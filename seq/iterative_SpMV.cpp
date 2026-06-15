//
// Sequential reference for iterative SpMV on an evolving sparse matrix.
//
// Evolution model:
//   The matrix is generated once. Every EPOCH_LEN iterations its rows are
//   logically shifted by shift_rows positions. The CSR data is never moved;
//   the SpMV kernel maps each logical row i to source row (i - row_shift) % n.
//
// Build:
//   g++ -O3 -std=c++20 -I ../common -Wall iterative_SpMV.cpp -o seq
//
// Usage:
//   ./seq -n N -nz K -m regular|irregular [-s seed] [--dump-vector FILE]
//

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "matrix_generation.hpp"
#include "utils.hpp"


static constexpr std::uint32_t NUM_ITERS = 500;
static constexpr std::uint32_t EPOCH_LEN = 25;  // iterations per epoch


static double dot(const std::vector<double>& a, const std::vector<double>& b) {
    return std::inner_product(a.begin(), a.end(), b.begin(), 0.0);
}

static double l2_norm(const std::vector<double>& x) {
    return std::sqrt(dot(x, x));
}

static void normalize(std::vector<double>& x) {
    const double nrm = l2_norm(x);
    const double inv = 1.0 / nrm;

    for (double& v : x) {
        v *= inv;
    }
}


static std::size_t compute_shift_rows(std::size_t n) {
    std::size_t s = n / 16 + 17;
    if ((s % 2) == 0) ++s;
    s %= n;
    if (s == 0) s = 1;
    return s;
}

// Per-row SpMV kernel.
//
// row_shift = s means that the matrix rows have been circularly shifted by s positions
//
// Logical row i uses source row (i - s mod n) from the original CSR matrix.
static void spmv_csr_shifted_rows(const CSRMatrix& A,
                                  std::size_t row_shift,
                                  const std::vector<double>& x,
                                  std::vector<double>& y) {
    const std::size_t n = A.n;
    y.assign(n, 0.0);

    for (std::size_t i = 0; i < n; ++i) {
        const std::size_t src_row = (i + n - row_shift) % n;

        double sum = 0.0;
        for (std::uint64_t p = A.row_ptr[src_row]; p < A.row_ptr[src_row + 1]; ++p) {
            sum += A.values[p] * x[A.col_idx[p]];
        }

        y[i] = sum;
    }
}

struct IterativeResult {
    double rayleigh             = 0.0;
    std::uint64_t checksum      = 0;
    std::size_t final_row_shift = 0;
};

static IterativeResult iterative_spmv_evolving(const CSRMatrix& A,
                                               std::uint64_t seed,
                                               std::vector<double>* final_vector = nullptr) {
    const std::size_t n = A.n;
    const std::size_t shift_rows = compute_shift_rows(n);

    // All parallel versions must reproduce this initialization exactly.
    std::vector<double> x(n);
    std::vector<double> y(n);

    SplitMix64 rng(seed ^ 0x123456789abcdef0ULL);
    for (double& v : x) {
        v = rng.next_unit();
    }
    normalize(x);

    // CSR is never modified; row_shift encodes the logical epoch state.
    std::size_t row_shift = 0;

    for (std::uint32_t iter = 0; iter < NUM_ITERS; ++iter) {
        // At each epoch boundary, update the logical row mapping.
        if (iter > 0 && (iter % EPOCH_LEN) == 0) {
            row_shift = (row_shift + shift_rows) % n;
        }

        // One iteration: shifted SpMV followed by vector normalization.
        // The normalization contains a global reduction.
        spmv_csr_shifted_rows(A, row_shift, x, y);
        normalize(y);

        x.swap(y);
    }

    // Extra SpMV outside the timed loop to compute the final Rayleigh value.
    spmv_csr_shifted_rows(A, row_shift, x, y);
    const double rayleigh = dot(x, y);
    const std::uint64_t checksum = checksum_vector(x);

    // Keep the final vector only if we have to dump it.
    if (final_vector != nullptr) {
        *final_vector = std::move(x);
    }

    return IterativeResult{
        .rayleigh = rayleigh,
        .checksum = checksum,
        .final_row_shift = row_shift
    };
}

int main(int argc, char** argv) {
    std::uint64_t n64  = 0;
    std::uint64_t nz   = 0;
    std::uint64_t seed = 111;
    std::string mode;
    std::string dump_vector_path;

    if (!read_arg_u64(argc, argv, "-n", n64) ||
        !read_arg_u64(argc, argv, "-nz", nz) ||
        !read_arg_str(argc, argv, "-m", mode)) {
        usage(argv[0]);
        return 1;
    }

    // Optional arguments
    (void)read_arg_u64(argc, argv, "-s", seed);
    (void)read_arg_str(argc, argv, "--dump-vector", dump_vector_path);

    const std::size_t n = static_cast<std::size_t>(n64);
    std::cout << "SPARSE_ITERATION_SEQ\n";

    try {
        const auto tg0 = std::chrono::steady_clock::now();
        const GeneratedMatrix G = generate_matrix(n, nz, seed, mode);
        const auto tg1 = std::chrono::steady_clock::now();

        const double generation_sec = std::chrono::duration<double>(tg1 - tg0).count();

        print_matrix_stats(G);
        std::cout << "generation_time_sec=" << generation_sec << "\n\n";

        std::vector<double>  final_vector;
        std::vector<double>* final_vector_out = dump_vector_path.empty() ? nullptr : &final_vector;

        const auto tc0 = std::chrono::steady_clock::now();
        const IterativeResult result = iterative_spmv_evolving(G.A, seed, final_vector_out);
        const auto tc1 = std::chrono::steady_clock::now();

        const double computation_sec = std::chrono::duration<double>(tc1 - tc0).count();

        std::cout << std::setprecision(15);
        std::cout << "rayleigh=" << result.rayleigh << "\n";
        std::cout << "checksum=0x" << std::hex << result.checksum << std::dec << "\n";

        std::cout << std::fixed << std::setprecision(6);
        std::cout << "Time (sec) = " << computation_sec << "\n";

        if (!dump_vector_path.empty()) {
            dump_vector(dump_vector_path, final_vector);
            std::cout << "vector_dump=" << dump_vector_path << "\n";
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
