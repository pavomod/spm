//
// OpenMP work-sharing implementation of iterative SpMV on an evolving sparse matrix.
//
// Build:
//   g++ -O3 -std=c++20 -Wall -Wextra -fopenmp -I ../common iterative_SpMV_omp_ws.cpp -o omp_ws
//
// Usage:
//   OMP_NUM_THREADS=8 ./omp_ws -n N -nz K -m regular|irregular [-t T] [-s seed] [-c chunk] [--dump-vector FILE]
//
//   -t T      override OMP_NUM_THREADS (optional)
//   -c K      chunk size for schedule(dynamic, K) (default: 64 rows)
//
// Synchronization structure (2 implicit barriers per iteration):
//   omp for schedule(dynamic, K) reduction(+:norm_sq)   → barrier 1 (end of for)
//   omp single { normalize; swap(x,y); epoch_for_next } → barrier 2 (end of single)
//
// Epoch update is folded into the normalize single: after iteration i, row_shift is
// updated for i+1 (semantically identical to updating at the START of i+1, since the
// barrier of the single runs before any thread begins i+1's SpMV).
//
// Comparison with omp_tasks: work-sharing avoids task creation/scheduling overhead
// (~O(n/K) task structs per iteration) but loses fine-grained task-stealing adaptivity.
// Benchmark both on spmcluster for irregular mode to quantify the trade-off.
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

static IterativeResult iterative_spmv_omp_ws(
    const CSRMatrix&     A,
    std::uint64_t        seed,
    int                  T,
    std::size_t          chunk_size,
    std::vector<double>* final_vector = nullptr)
{
    const std::size_t n          = A.n;
    const std::size_t shift_rows = compute_shift_rows(n);
    const int         chunk      = static_cast<int>(chunk_size);

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
    double norm_sq = 0.0;  // shared; reset to 0 each iter via omp single

    #pragma omp parallel num_threads(T)
    {
        for (std::uint32_t iter = 0; iter < NUM_ITERS; ++iter) {

            // Reset shared accumulator before each reduction.
            // omp single has implicit barrier → all threads see norm_sq=0 before omp for.
            #pragma omp single
            norm_sq = 0.0;

            // schedule(dynamic, chunk): each thread steals 'chunk' rows at a time.
            // reduction(+:norm_sq): private copies init to 0, summed into norm_sq at barrier.
            #pragma omp for schedule(dynamic, chunk) reduction(+:norm_sq)
            for (std::size_t i = 0; i < n; ++i) {
                const std::size_t src = (i + n - row_shift) % n;
                double yi = 0.0;
                for (std::uint64_t p = A.row_ptr[src]; p < A.row_ptr[src + 1]; ++p)
                    yi += A.values[p] * x[A.col_idx[p]];
                y[i]    = yi;
                norm_sq += yi * yi;
            }
            // Implicit barrier: all y[i] written, norm_sq holds global ||y||².

            // Normalize y, swap buffers, then pre-compute epoch shift for the next iteration.
            // Pre-computing here (at end of iter i) is equivalent to the sequential code
            // computing it at the start of iter i+1: both happen after the barrier of iter i
            // and before any thread begins the SpMV of iter i+1.
            #pragma omp single
            {
                const double inv = 1.0 / std::sqrt(norm_sq);
                for (std::size_t i = 0; i < n; ++i) y[i] *= inv;
                std::swap(x, y);

                const std::uint32_t next = iter + 1;
                if (next > 0 && next < NUM_ITERS && (next % EPOCH_LEN) == 0)
                    row_shift = (row_shift + shift_rows) % n;
            }
            // Implicit barrier: x is normalized, row_shift ready for next iteration.
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

    std::cout << "SPARSE_ITERATION_OMP_WS\n";

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
            iterative_spmv_omp_ws(G.A, seed, T, K, fv_out);
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
