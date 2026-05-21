//
// MPI + OpenMP implementation of iterative SpMV on an evolving sparse matrix.
//
// Build:
//   mpicxx -O3 -std=c++20 -fopenmp -I ../common iterative_SpMV_mpi.cpp -o mpi_omp
//
// Usage:
//   OMP_NUM_THREADS=4 mpirun -np 4 ./mpi_omp -n N -nz K -m regular|irregular
//                            [-t T] [-s seed] [-c chunk] [--dump-vector FILE]
//
// Distribution strategy (AD-002 Option A — logical mapping, no epoch redistribution):
//   Physical rows partitioned nnz-balanced across P ranks once at startup (fixed forever).
//   row_shift advances at epoch boundaries via local arithmetic; no MPI communication.
//   SpMV: rank r computes dot products for its physical rows [phys_start, phys_end).
//   y_local[j] is the result for LOGICAL row (phys_start + j + row_shift) % n.
//
// Per-iteration communication:
//   1. MPI_Allreduce — global ||y||² → inv_norm  (O(1) message, latency-bound)
//   2. MPI_Allgatherv — collect y_local into phys_buf[n]  (O(n) data)
//   3. cyclic_copy   — phys_buf → x: 2 memcpy calls  (O(n), no modular arithmetic)
//
// x vector: fully replicated on all ranks (AD-003).
// Only rank 0 prints output.
//

#include <mpi.h>
#include <omp.h>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "matrix_generation.hpp"
#include "utils.hpp"

static constexpr std::uint32_t NUM_ITERS = 500;
static constexpr std::uint32_t EPOCH_LEN = 25;

static std::size_t compute_shift_rows(std::size_t n) {
    std::size_t s = n / 16 + 17;
    if ((s % 2) == 0) ++s;
    s %= n;
    if (s == 0) s = 1;
    return s;
}

// nnz-balanced partition. Returns part[0..P]: rank r owns physical rows [part[r], part[r+1]).
static std::vector<int> compute_partition(const std::vector<std::uint64_t>& row_ptr, int P) {
    std::vector<int> part(static_cast<std::size_t>(P) + 1);
    part[0] = 0;
    const std::uint64_t total = row_ptr.back();
    for (int r = 1; r < P; ++r) {
        const std::uint64_t target =
            static_cast<std::uint64_t>(r) * total / static_cast<std::uint64_t>(P);
        const auto it = std::lower_bound(row_ptr.begin(), row_ptr.end(), target);
        part[static_cast<std::size_t>(r)] =
            std::min(static_cast<int>(it - row_ptr.begin()),
                     static_cast<int>(row_ptr.size()) - 1);
    }
    part[static_cast<std::size_t>(P)] = static_cast<int>(row_ptr.size()) - 1;
    return part;
}

// dst[i] = src[(i + n - shift) % n]  (cyclic right-shift of src by shift positions)
// Implemented as two contiguous memcpy: no per-element modular arithmetic.
static void cyclic_copy(const std::vector<double>& src,
                        std::size_t shift, std::size_t n,
                        std::vector<double>& dst) {
    if (shift == 0) {
        std::copy(src.cbegin(), src.cend(), dst.begin());
        return;
    }
    // dst[0..shift-1]   = src[n-shift..n-1]
    // dst[shift..n-1]   = src[0..n-shift-1]
    const auto pivot = static_cast<std::ptrdiff_t>(n - shift);
    std::copy(src.cbegin() + pivot, src.cend(), dst.begin());
    std::copy(src.cbegin(), src.cbegin() + pivot,
              dst.begin() + static_cast<std::ptrdiff_t>(shift));
}

int main(int argc, char** argv) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int my_rank, P;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &P);

    // --- Argument parsing (all ranks) ---
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
        if (my_rank == 0)
            std::cerr << "Usage: mpirun -np P " << argv[0]
                      << " -n N -nz K -m regular|irregular"
                         " [-t T] [-s seed] [-c chunk] [--dump-vector FILE]\n";
        MPI_Finalize();
        return 1;
    }
    (void)read_arg_u64(argc, argv, "-s",  seed);
    (void)read_arg_u64(argc, argv, "-c",  chunk64);
    (void)read_arg_str(argc, argv, "--dump-vector", dump_path);
    if (read_arg_u64(argc, argv, "-t", t64) && t64 > 0)
        omp_set_num_threads(static_cast<int>(t64));

    const std::size_t n = static_cast<std::size_t>(n64);
    const int         T = omp_get_max_threads();
    const int         K = static_cast<int>(std::max<std::uint64_t>(1, chunk64));

    if (my_rank == 0)
        std::cout << "SPARSE_ITERATION_MPI_OMP\n";

    // Portable MPI datatypes — guaranteed on Linux x86_64
    static_assert(sizeof(std::uint64_t) == sizeof(unsigned long long));
    static_assert(sizeof(std::uint32_t) == sizeof(unsigned int));
    const MPI_Datatype MPI_UINT64 = MPI_UNSIGNED_LONG_LONG;
    const MPI_Datatype MPI_UINT32 = MPI_UNSIGNED;

    // --- Matrix generation (rank 0) and distribution ---
    //
    // G is generated only on rank 0. Non-root ranks have an empty GeneratedMatrix
    // (default-constructed). MPI ignores sendbuf on non-root processes.
    //
    GeneratedMatrix G;
    std::vector<int> part(static_cast<std::size_t>(P) + 1, 0);

    // Sendcount/displacement arrays — allocated only on rank 0.
    std::vector<int> rp_sc, rp_dp;   // row_ptr scatter
    std::vector<int> dt_sc, dt_dp;   // col_idx / values scatter

    std::uint64_t total_nnz_g   = 0;
    std::uint32_t min_nnz_row_g = 0;
    std::uint32_t max_nnz_row_g = 0;

    const double t_gen0 = MPI_Wtime();

    if (my_rank == 0) {
        G     = generate_matrix(n, nz, seed, mode);
        part  = compute_partition(G.A.row_ptr, P);

        total_nnz_g   = G.stats.total_nnz;
        min_nnz_row_g = G.stats.min_nnz_per_row;
        max_nnz_row_g = G.stats.max_nnz_per_row;

        rp_sc.resize(static_cast<std::size_t>(P));
        rp_dp.resize(static_cast<std::size_t>(P));
        dt_sc.resize(static_cast<std::size_t>(P));
        dt_dp.resize(static_cast<std::size_t>(P));

        for (int r = 0; r < P; ++r) {
            rp_sc[static_cast<std::size_t>(r)] = part[static_cast<std::size_t>(r) + 1]
                                                - part[static_cast<std::size_t>(r)] + 1;
            rp_dp[static_cast<std::size_t>(r)] = part[static_cast<std::size_t>(r)];

            dt_sc[static_cast<std::size_t>(r)] =
                static_cast<int>(G.A.row_ptr[static_cast<std::size_t>(part[static_cast<std::size_t>(r) + 1])]
                               - G.A.row_ptr[static_cast<std::size_t>(part[static_cast<std::size_t>(r)])]);
            dt_dp[static_cast<std::size_t>(r)] =
                static_cast<int>(G.A.row_ptr[static_cast<std::size_t>(part[static_cast<std::size_t>(r)])]);
        }
    }

    // Broadcast partition and scalar stats to all ranks
    MPI_Bcast(part.data(), P + 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&total_nnz_g, 1, MPI_UINT64, 0, MPI_COMM_WORLD);

    const int local_n = part[static_cast<std::size_t>(my_rank) + 1]
                      - part[static_cast<std::size_t>(my_rank)];

    // Scatter row_ptr
    std::vector<std::uint64_t> local_row_ptr(static_cast<std::size_t>(local_n) + 1);
    MPI_Scatterv(my_rank == 0 ? G.A.row_ptr.data() : nullptr,
                 my_rank == 0 ? rp_sc.data() : nullptr,
                 my_rank == 0 ? rp_dp.data() : nullptr,
                 MPI_UINT64,
                 local_row_ptr.data(), local_n + 1, MPI_UINT64,
                 0, MPI_COMM_WORLD);

    // Derive local nnz from raw (global-offset) row_ptr, then make 0-based
    const std::uint64_t rp_base = local_row_ptr[0];
    const int local_nnz =
        static_cast<int>(local_row_ptr[static_cast<std::size_t>(local_n)] - rp_base);
    for (auto& p : local_row_ptr) p -= rp_base;

    // Scatter col_idx and values
    std::vector<std::uint32_t> local_col_idx(static_cast<std::size_t>(local_nnz));
    std::vector<double>        local_values(static_cast<std::size_t>(local_nnz));

    MPI_Scatterv(my_rank == 0 ? G.A.col_idx.data() : nullptr,
                 my_rank == 0 ? dt_sc.data() : nullptr,
                 my_rank == 0 ? dt_dp.data() : nullptr,
                 MPI_UINT32,
                 local_col_idx.data(), local_nnz, MPI_UINT32,
                 0, MPI_COMM_WORLD);
    MPI_Scatterv(my_rank == 0 ? G.A.values.data() : nullptr,
                 my_rank == 0 ? dt_sc.data() : nullptr,
                 my_rank == 0 ? dt_dp.data() : nullptr,
                 MPI_DOUBLE,
                 local_values.data(), local_nnz, MPI_DOUBLE,
                 0, MPI_COMM_WORLD);

    // Free full matrix on rank 0 (no longer needed; saves memory during computation)
    if (my_rank == 0) G = GeneratedMatrix{};

    const double gen_sec = MPI_Wtime() - t_gen0;

    // --- Print header (rank 0 only) ---
    if (my_rank == 0) {
        std::cout << "rows=" << n
                  << "  total_nnz=" << total_nnz_g
                  << "  min_nnz_per_row=" << min_nnz_row_g
                  << "  max_nnz_per_row=" << max_nnz_row_g << "  ";
        std::cout << std::fixed << std::setprecision(3);
        std::cout << "avg_nnz_per_row="
                  << static_cast<double>(total_nnz_g) / static_cast<double>(n) << "\n";
        std::cout << "generation_time_sec=" << gen_sec << "\n";
        std::cout << "num_procs=" << P
                  << "  num_threads_per_proc=" << T << "\n\n";
    }

    // --- Pre-compute Allgatherv parameters (fixed for all iterations) ---
    std::vector<int> ag_recvcounts(static_cast<std::size_t>(P));
    std::vector<int> ag_displs(static_cast<std::size_t>(P));
    for (int r = 0; r < P; ++r) {
        ag_recvcounts[static_cast<std::size_t>(r)] =
            part[static_cast<std::size_t>(r) + 1] - part[static_cast<std::size_t>(r)];
        ag_displs[static_cast<std::size_t>(r)] = part[static_cast<std::size_t>(r)];
    }

    // --- Initialize replicated x (identical on all ranks — same seed, same RNG) ---
    std::vector<double> x(n), phys_buf(n), y_local(static_cast<std::size_t>(local_n));
    {
        SplitMix64 rng(seed ^ 0x123456789abcdef0ULL);
        for (double& v : x) v = rng.next_unit();
        double sq = 0.0;
        for (double v : x) sq += v * v;
        const double inv = 1.0 / std::sqrt(sq);
        for (double& v : x) v *= inv;
    }

    // --- Timed iterative computation ---
    const std::size_t shift_rows = compute_shift_rows(n);
    std::size_t row_shift = 0;

    double t_compute   = 0.0;
    double t_allreduce = 0.0;
    double t_allgather = 0.0;
    double t_epoch     = 0.0;

    MPI_Barrier(MPI_COMM_WORLD);
    const double tc0 = MPI_Wtime();

    for (std::uint32_t iter = 0; iter < NUM_ITERS; ++iter) {

        // Epoch update: local arithmetic, no communication.
        // row_shift maps physical row phys_j to logical row (phys_j + row_shift) % n.
        {
            const double te0 = MPI_Wtime();
            if (iter > 0 && (iter % EPOCH_LEN) == 0)
                row_shift = (row_shift + shift_rows) % n;
            t_epoch += MPI_Wtime() - te0;
        }

        // SpMV: each thread handles K physical rows.
        // y_local[j] = A_local[j, :] · x  (no row_shift in hot loop).
        double local_sq = 0.0;
        {
            const double tk0 = MPI_Wtime();
            #pragma omp parallel for schedule(dynamic, K) num_threads(T) \
                                     reduction(+:local_sq)
            for (int j = 0; j < local_n; ++j) {
                double val = 0.0;
                for (std::uint64_t p = local_row_ptr[static_cast<std::size_t>(j)];
                     p < local_row_ptr[static_cast<std::size_t>(j) + 1]; ++p)
                    val += local_values[p] * x[local_col_idx[p]];
                y_local[static_cast<std::size_t>(j)] = val;
                local_sq += val * val;
            }
            t_compute += MPI_Wtime() - tk0;
        }

        // Global norm reduction
        {
            const double tr0 = MPI_Wtime();
            double global_sq = 0.0;
            MPI_Allreduce(&local_sq, &global_sq, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            t_allreduce += MPI_Wtime() - tr0;

            const double inv = 1.0 / std::sqrt(global_sq);
            for (int j = 0; j < local_n; ++j)
                y_local[static_cast<std::size_t>(j)] *= inv;
        }

        // x assembly:
        //   Allgatherv → phys_buf[phys_j] = normalized y for logical row (phys_j + row_shift) % n
        //   cyclic_copy → x[i] = phys_buf[(i + n - row_shift) % n]
        //   (two contiguous memcpy calls, O(n), sequential, cache-friendly)
        {
            const double tg0 = MPI_Wtime();
            MPI_Allgatherv(y_local.data(), local_n, MPI_DOUBLE,
                           phys_buf.data(),
                           ag_recvcounts.data(), ag_displs.data(),
                           MPI_DOUBLE, MPI_COMM_WORLD);
            cyclic_copy(phys_buf, row_shift, n, x);
            t_allgather += MPI_Wtime() - tg0;
        }
    }

    const double total_sec = MPI_Wtime() - tc0;

    // --- Final rayleigh (outside timed region) ---
    // Each rank contributes its partial dot x · (A_local · x).
    double local_ray = 0.0;
    for (int j = 0; j < local_n; ++j) {
        double val = 0.0;
        for (std::uint64_t p = local_row_ptr[static_cast<std::size_t>(j)];
             p < local_row_ptr[static_cast<std::size_t>(j) + 1]; ++p)
            val += local_values[p] * x[local_col_idx[p]];
        // This val belongs at logical position (phys_start + j + row_shift) % n
        const std::size_t phys_j   = static_cast<std::size_t>(part[static_cast<std::size_t>(my_rank)])
                                   + static_cast<std::size_t>(j);
        const std::size_t logical_i = (phys_j + row_shift) % n;
        local_ray += val * x[logical_i];
    }
    double rayleigh = 0.0;
    MPI_Allreduce(&local_ray, &rayleigh, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    // Checksum: x is replicated, all ranks would get the same value.
    const std::uint64_t cksum = (my_rank == 0) ? checksum_vector(x) : 0;

    // --- Output (rank 0 only) ---
    if (my_rank == 0) {
        std::cout << std::setprecision(15);
        std::cout << "rayleigh=" << rayleigh << "\n";
        std::cout << "checksum=0x" << std::hex << cksum << std::dec << "\n";
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "Time (sec) = "             << total_sec    << "\n";
        std::cout << "local_compute_time_sec="   << t_compute    << "\n";
        std::cout << "communication_time_sec="   << t_allgather  << "\n";
        std::cout << "reduction_time_sec="       << t_allreduce  << "\n";
        std::cout << "epoch_transition_time_sec=" << t_epoch     << "\n";

        if (!dump_path.empty()) {
            dump_vector(dump_path, x);
            std::cout << "vector_dump=" << dump_path << "\n";
        }
    }

    MPI_Finalize();
    return 0;
}
