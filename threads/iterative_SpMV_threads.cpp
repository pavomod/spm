//
// C++ threads implementation of iterative SpMV on an evolving sparse matrix.
//
// Build:
//   g++ -O3 -std=c++20 -Wall -Wextra -pthread -I ../common iterative_SpMV_threads.cpp -o threads
//
// Usage:
//   ./threads -n N -nz K -m regular|irregular [-t T] [-s seed] [-d] [-c chunk] [--dump-vector FILE]
//
//   -t T      number of threads (default: hardware_concurrency)
//   -d        dynamic chunk scheduling (default: static nnz-balanced partitioning)
//   -c K      chunk size in rows for dynamic mode (default: 64)
//

#include <atomic>
#include <barrier>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
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

// Assigns roughly equal total nnz to each thread using binary search on row_ptr.
// Returns partition[0..T]: thread t owns rows [partition[t], partition[t+1]).
static std::vector<std::size_t> compute_nnz_partition(const CSRMatrix& A, int T) {
    std::vector<std::size_t> part(static_cast<std::size_t>(T) + 1);
    part[0] = 0;
    const std::uint64_t total = A.row_ptr[A.n];
    for (int t = 1; t < T; ++t) {
        const std::uint64_t target = static_cast<std::uint64_t>(t) * total
                                     / static_cast<std::uint64_t>(T);
        auto it = std::lower_bound(A.row_ptr.begin(), A.row_ptr.end(), target);
        const std::size_t r = static_cast<std::size_t>(it - A.row_ptr.begin());
        part[static_cast<std::size_t>(t)] = std::min(r, A.n);
    }
    part[static_cast<std::size_t>(T)] = A.n;
    return part;
}

struct IterativeResult {
    double        rayleigh        = 0.0;
    std::uint64_t checksum        = 0;
    std::size_t   final_row_shift = 0;
};

static IterativeResult iterative_spmv_threads(
    const CSRMatrix&     A,
    std::uint64_t        seed,
    int                  T,
    bool                 dynamic_sched,
    std::size_t          chunk_size,
    std::vector<double>* final_vector = nullptr)
{
    const std::size_t n          = A.n;
    const std::size_t shift_rows = compute_shift_rows(n);

    // Double-buffered x/y — pointers are swapped each iteration.
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

    std::size_t   row_shift    = 0;
    std::uint32_t current_iter = 0;

    std::vector<PaddedDouble>      partial_norms(static_cast<std::size_t>(T));
    std::atomic<std::size_t>       next_chunk{0};
    const std::vector<std::size_t> partition = compute_nnz_partition(A, T);

    // Barrier completion — runs exactly once per phase while all threads are blocked.
    // Performs: epoch advance, global norm reduction, serial scale of y, pointer swap.
    // Serial scaling is a minor fraction of total work for large n (O(n) vs O(nnz)).
    auto completion = [&]() noexcept {
        if (current_iter > 0 && (current_iter % EPOCH_LEN) == 0)
            row_shift = (row_shift + shift_rows) % n;

        double total = 0.0;
        for (int t = 0; t < T; ++t)
            total += partial_norms[static_cast<std::size_t>(t)].val;
        const double inv = 1.0 / std::sqrt(total);

        for (std::size_t i = 0; i < n; ++i) y[i] *= inv;

        std::swap(x, y);
        next_chunk.store(0, std::memory_order_relaxed);
        ++current_iter;
    };

    std::barrier<decltype(completion)> sync(T, std::move(completion));

    // Worker: each thread runs this. tid 0 is the calling (main) thread.
    auto worker = [&](int tid) {
        while (current_iter < NUM_ITERS) {
            double psum = 0.0;

            if (dynamic_sched) {
                // Steal chunks from shared counter — handles irregular load imbalance.
                while (true) {
                    const std::size_t cs =
                        next_chunk.fetch_add(chunk_size, std::memory_order_relaxed);
                    if (cs >= n) break;
                    const std::size_t ce = std::min(cs + chunk_size, n);
                    for (std::size_t i = cs; i < ce; ++i) {
                        const std::size_t src = (i + n - row_shift) % n;
                        double yi = 0.0;
                        for (std::uint64_t p = A.row_ptr[src]; p < A.row_ptr[src + 1]; ++p)
                            yi += A.values[p] * x[A.col_idx[p]];
                        y[i] = yi;
                        psum += yi * yi;
                    }
                }
            } else {
                // Static nnz-balanced partition — near-perfect balance for irregular mode
                // because work is proportional to nnz, and partition equalises nnz.
                const std::size_t row_begin = partition[static_cast<std::size_t>(tid)];
                const std::size_t row_end   = partition[static_cast<std::size_t>(tid) + 1];
                for (std::size_t i = row_begin; i < row_end; ++i) {
                    const std::size_t src = (i + n - row_shift) % n;
                    double yi = 0.0;
                    for (std::uint64_t p = A.row_ptr[src]; p < A.row_ptr[src + 1]; ++p)
                        yi += A.values[p] * x[A.col_idx[p]];
                    y[i] = yi;
                    psum += yi * yi;
                }
            }

            partial_norms[static_cast<std::size_t>(tid)].val = psum;
            sync.arrive_and_wait();   // single synchronization point per iteration
        }
    };

    std::vector<std::thread> workers;
    workers.reserve(static_cast<std::size_t>(T - 1));
    for (int t = 1; t < T; ++t)
        workers.emplace_back(worker, t);
    worker(0);
    for (auto& w : workers) w.join();

    // Final diagnostics (outside timed region) — sequential to match seq reference exactly.
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
    const std::uint64_t       cksum = checksum_vector(x_vec);

    if (final_vector != nullptr)
        *final_vector = x_vec;

    return {rayleigh, cksum, row_shift};
}

int main(int argc, char** argv) {
    std::uint64_t n64     = 0;
    std::uint64_t nz      = 0;
    std::uint64_t seed    = 111;
    std::uint64_t t64     = std::thread::hardware_concurrency();
    std::uint64_t chunk64 = 64;
    std::string   mode;
    std::string   dump_path;
    bool          dynamic_sched = false;

    if (!read_arg_u64(argc, argv, "-n",  n64)  ||
        !read_arg_u64(argc, argv, "-nz", nz)   ||
        !read_arg_str(argc, argv, "-m",  mode)) {
        std::cerr << "Usage: " << argv[0]
                  << " -n N -nz K -m regular|irregular"
                     " [-t T] [-s seed] [-d] [-c chunk] [--dump-vector FILE]\n";
        return 1;
    }
    (void)read_arg_u64(argc, argv, "-s",  seed);
    (void)read_arg_u64(argc, argv, "-t",  t64);
    (void)read_arg_u64(argc, argv, "-c",  chunk64);
    (void)read_arg_str(argc, argv, "--dump-vector", dump_path);
    for (int i = 1; i < argc; ++i)
        if (std::string(argv[i]) == "-d") { dynamic_sched = true; break; }

    const std::size_t n = static_cast<std::size_t>(n64);
    const int         T = static_cast<int>(std::max<std::uint64_t>(1, t64));
    const std::size_t K = static_cast<std::size_t>(std::max<std::uint64_t>(1, chunk64));

    std::cout << "SPARSE_ITERATION_THREADS\n";

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
            iterative_spmv_threads(G.A, seed, T, dynamic_sched, K, fv_out);
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
