//
// OpenMP task-based implementation of iterative SpMV on an evolving sparse matrix.
//
// Build:
//   g++ -O3 -std=c++20 -Wall -Wextra -fopenmp -I ../common iterative_SpMV_omp_tasks.cpp -o omp_tasks
//
// Usage:
//   OMP_NUM_THREADS=8 ./omp_tasks -n N -nz K -m regular|irregular [-t T] [-s seed] [-c chunk] [--dump-vector FILE]
//
//   -t T      override OMP_NUM_THREADS (optional)
//   -c K      rows per task / chunk size (default: 64); tune K for irregular mode
//
// Synchronization structure (2 implicit barriers per iteration):
//   single { epoch_update; create tasks; taskwait } → barrier 1
//   single { reduce norm; scale y; swap x↔y }       → barrier 2
//
// While the single thread waits in taskwait, other threads execute tasks at the
// implicit barrier of the single construct (a task-scheduling point per OpenMP spec).
//

#include <omp.h>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "matrix_generation.hpp"
#include "utils.hpp"

static constexpr std::uint32_t NUM_ITERS  = 500;
static constexpr std::uint32_t EPOCH_LEN  = 25;
static constexpr std::size_t   CACHE_LINE = 64;

// Padded accumulator — prevents false sharing on partial norm sums
struct alignas(CACHE_LINE) PaddedDouble {
    double val = 0.0;
};

static std::size_t compute_shift_rows(std::size_t n) {
    std::size_t s = n / 16 + 17;
    if ((s % 2) == 0) ++s;
    s %= n;
    if (s == 0) s = 1;
    return s;
}

struct IterativeResult {
    double        rayleigh        = 0.0;
    std::uint64_t checksum        = 0;
    std::size_t   final_row_shift = 0;
};

static IterativeResult iterative_spmv_omp_tasks(
    const CSRMatrix&     A,
    std::uint64_t        seed,
    int                  T,
    std::size_t          chunk_size,
    std::vector<double>* final_vector = nullptr)
{
    const std::size_t n          = A.n;
    const std::size_t shift_rows = compute_shift_rows(n);

    // Double-buffered x/y — pointer swap each iteration, no data copies
    std::vector<double> buf0(n), buf1(n);
    {
        SplitMix64 rng(seed ^ 0x123456789abcdef0ULL);
        for (double& v : buf0) v = rng.next_unit();
        double s = 0.0;
        for (double v : buf0) s += v * v;
        const double inv = 1.0 / std::sqrt(s);
        for (double& v : buf0) v *= inv;
    }

    double* x = buf0.data();   // read in SpMV
    double* y = buf1.data();   // written in SpMV

    std::size_t row_shift = 0;

    // Per-thread partial norm accumulators; reset each iteration in the second single.
    std::vector<PaddedDouble> partial_norms(static_cast<std::size_t>(T));

    #pragma omp parallel num_threads(T)
    {
        for (std::uint32_t iter = 0; iter < NUM_ITERS; ++iter) {

            // Phase 1: epoch update + SpMV task creation.
            //
            // The single thread creates all tasks and calls taskwait.
            // Other threads waiting at this single's implicit barrier execute the
            // tasks (barriers are task-scheduling points in the OpenMP spec).
            // When taskwait returns, all tasks are done; then the single thread
            // reaches the implicit barrier and all threads synchronize.
            #pragma omp single
            {
                if (iter > 0 && (iter % EPOCH_LEN) == 0)
                    row_shift = (row_shift + shift_rows) % n;

                for (std::size_t cs = 0; cs < n; cs += chunk_size) {
                    const std::size_t ce = std::min(cs + chunk_size, n);

                    #pragma omp task firstprivate(cs, ce)
                    {
                        const int         tid  = omp_get_thread_num();
                        const std::size_t  tid_ = static_cast<std::size_t>(tid);
                        double psum = 0.0;

                        for (std::size_t i = cs; i < ce; ++i) {
                            const std::size_t src = (i + n - row_shift) % n;
                            double yi = 0.0;
                            for (std::uint64_t p = A.row_ptr[src];
                                 p < A.row_ptr[src + 1]; ++p)
                                yi += A.values[p] * x[A.col_idx[p]];
                            y[i]  = yi;
                            psum += yi * yi;
                        }

                        // Safe without atomics: a thread executes one task at a time
                        // (tasks have no suspension points here, so no interleaving).
                        partial_norms[tid_].val += psum;
                    }
                }

                #pragma omp taskwait   // wait for all child tasks before leaving single
            }
            // Implicit barrier: all tasks complete, all threads synchronized.

            // Phase 2: reduce partial norms, scale y, swap buffers.
            // Serial O(n) scale is small relative to O(nnz) SpMV for large n.
            #pragma omp single
            {
                double total = 0.0;
                for (int t = 0; t < T; ++t) {
                    const std::size_t t_ = static_cast<std::size_t>(t);
                    total           += partial_norms[t_].val;
                    partial_norms[t_].val = 0.0;   // reset for next iteration
                }
                const double inv = 1.0 / std::sqrt(total);
                for (std::size_t i = 0; i < n; ++i) y[i] *= inv;
                std::swap(x, y);
            }
            // Implicit barrier: x is normalized and ready for next iteration.
        }
    }

    // Final diagnostics (outside timed region) — sequential SpMV, matches seq reference.
    std::vector<double> y_final(n);
    for (std::size_t i = 0; i < n; ++i) {
        const std::size_t src = (i + n - row_shift) % n;
        double yi = 0.0;
        for (std::uint64_t p = A.row_ptr[src]; p < A.row_ptr[src + 1]; ++p)
            yi += A.values[p] * x[A.col_idx[p]];
        y_final[i] = yi;
    }
    const double rayleigh = std::inner_product(x, x + n, y_final.data(), 0.0);

    const std::vector<double> x_vec(x, x + n);
    const std::uint64_t cksum = checksum_vector(x_vec);

    if (final_vector != nullptr) *final_vector = x_vec;

    return {rayleigh, cksum, row_shift};
}

int main(int argc, char** argv) {
    std::uint64_t n64     = 0;
    std::uint64_t nz      = 0;
    std::uint64_t seed    = 111;
    std::uint64_t chunk64 = 64;
    std::uint64_t t64     = 0;
    std::string   mode;
    std::string   dump_path;

    if (!read_arg_u64(argc, argv, "-n",  n64) ||
        !read_arg_u64(argc, argv, "-nz", nz)  ||
        !read_arg_str(argc, argv, "-m",  mode)) {
        std::cerr << "Usage: " << argv[0]
                  << " -n N -nz K -m regular|irregular"
                     " [-t T] [-s seed] [-c chunk] [--dump-vector FILE]\n";
        return 1;
    }
    (void)read_arg_u64(argc, argv, "-s",  seed);
    (void)read_arg_u64(argc, argv, "-c",  chunk64);
    (void)read_arg_str(argc, argv, "--dump-vector", dump_path);

    if (read_arg_u64(argc, argv, "-t", t64) && t64 > 0)
        omp_set_num_threads(static_cast<int>(t64));

    const std::size_t n = static_cast<std::size_t>(n64);
    const std::size_t K = static_cast<std::size_t>(std::max<std::uint64_t>(1, chunk64));
    const int         T = omp_get_max_threads();

    std::cout << "SPARSE_ITERATION_OMP_TASKS\n";

    try {
        const auto tg0 = std::chrono::steady_clock::now();
        const GeneratedMatrix G = generate_matrix(n, nz, seed, mode);
        const auto tg1 = std::chrono::steady_clock::now();
        const double gen_sec = std::chrono::duration<double>(tg1 - tg0).count();

        print_matrix_stats(G);
        std::cout << "generation_time_sec=" << gen_sec << "\n";
        std::cout << "num_threads=" << T << "\n\n";

        std::vector<double>  final_vec;
        std::vector<double>* fv_out = dump_path.empty() ? nullptr : &final_vec;

        const auto tc0 = std::chrono::steady_clock::now();
        const IterativeResult res =
            iterative_spmv_omp_tasks(G.A, seed, T, K, fv_out);
        const auto tc1 = std::chrono::steady_clock::now();
        const double comp_sec = std::chrono::duration<double>(tc1 - tc0).count();

        std::cout << std::setprecision(15);
        std::cout << "rayleigh=" << res.rayleigh << "\n";
        std::cout << "checksum=0x" << std::hex << res.checksum << std::dec << "\n";
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "Time (sec) = " << comp_sec << "\n";

        if (!dump_path.empty()) {
            dump_vector(dump_path, final_vec);
            std::cout << "vector_dump=" << dump_path << "\n";
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
