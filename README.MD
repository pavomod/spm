# Iterative SpMV — Parallel Implementations

Iterative Sparse Matrix-Vector computation on an evolving sparse matrix.
Sequential reference + C++ threads + OpenMP tasks + MPI+OpenMP.

---

## Build

```bash
# All at once
make all

# Individual modules
make -C seq
make -C threads
make -C omp_tasks
make -C mpi_omp

# Clean
make clean
```

Requires: `g++ >= 10` (C++20 for `std::barrier`), `libopenmpi-dev` for MPI target.

---

## Run

```bash
# Sequential
./seq/seq -n 500000 -nz 20000000 -m irregular

# C++ Threads — static nnz-balanced partition, 8 threads
./threads/threads -n 500000 -nz 20000000 -m irregular -t 8

# C++ Threads — dynamic chunk scheduling, 8 threads, chunk=64 rows
./threads/threads -n 500000 -nz 20000000 -m irregular -t 8 -d

# OpenMP Tasks — 8 threads
OMP_NUM_THREADS=8 ./omp_tasks/omp_tasks -n 500000 -nz 20000000 -m irregular

# MPI + OpenMP — 4 procs x 4 threads
OMP_NUM_THREADS=4 mpirun -np 4 ./mpi_omp/mpi_omp -n 500000 -nz 20000000 -m irregular
```

---

## Correctness Check

All parallel versions must produce a Rayleigh value within relative tolerance `1e-10` of the sequential reference.

```bash
# Step 1: generate reference dump
./seq/seq -n 50000 -nz 200000 -m irregular --dump-vector /tmp/seq.dump

# Step 2: generate parallel dump
./threads/threads -n 50000 -nz 200000 -m irregular -t 4 --dump-vector /tmp/thr.dump

# Step 3: element-wise diff (no output = PASS)
awk 'NR==FNR{a[NR]=$1;next} {d=a[FNR]-$1; if(d<0)d=-d; if(d>1e-10) print NR, d}' \
    /tmp/seq.dump /tmp/thr.dump
```

For MPI:
```bash
OMP_NUM_THREADS=4 mpirun -np 4 ./mpi_omp/mpi_omp -n 50000 -nz 200000 -m irregular \
    --dump-vector /tmp/mpi.dump
awk 'NR==FNR{a[NR]=$1;next} {d=a[FNR]-$1; if(d<0)d=-d; if(d>1e-10) print NR, d}' \
    /tmp/seq.dump /tmp/mpi.dump
```

---

## Test Sizes

| Label | n | nz | mode | Use |
|-------|---|----|------|-----|
| smoke | 5,000 | 20,000 | irregular | quick build check |
| correctness | 50,000 | 200,000 | irregular | element-wise diff |
| perf-med | 500,000 | 20,000,000 | irregular | single-node benchmark |
| perf-large | 2,000,000 | 80,000,000 | irregular | MPI scaling |

---

## Output Format

```
SPARSE_ITERATION_[SEQ|THREADS|OMP_TASKS|MPI_OMP]
rows=N  total_nnz=K  min_nnz_per_row=...  max_nnz_per_row=...  avg_nnz_per_row=...
generation_time_sec=...
[num_threads=T]          <- parallel versions only

rayleigh=<15 sig digits>
checksum=0x<hex>
Time (sec) = <6 decimal places>
```

---

## CLI Flags

| Flag | Applies to | Default |
|------|-----------|---------|
| `-n N` | all | required |
| `-nz K` | all | required |
| `-m regular\|irregular` | all | required |
| `-s seed` | all | 111 |
| `--dump-vector FILE` | all | off |
| `-t T` | threads | hardware_concurrency |
| `-d` | threads | off (static mode) |
| `-c K` | threads | 64 rows |
