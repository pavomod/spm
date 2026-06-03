#!/bin/bash
BIN=~/gmodica5/omp_tasks/omp_tasks
N=500000
NZ=20000000
MODE=irregular
OUTDIR=~/gmodica5/results/omp_tasks
mkdir -p $OUTDIR

for T in 1 2 4 8 16 32; do
    echo "=== T=$T, K=64 ==="
    for RUN in 1 2 3; do
        OMP_PROC_BIND=close OMP_PLACES=cores OMP_NUM_THREADS=$T numactl --interleave=all $BIN -n $N -nz $NZ -m $MODE -c 64 2>&1 | \
            tee $OUTDIR/n500k_T${T}_K64_run${RUN}.txt
    done
done

T=16
for K in 16 64 256 1024; do
    echo "=== T=$T, K=$K ==="
    for RUN in 1 2 3; do
        OMP_PROC_BIND=close OMP_PLACES=cores OMP_NUM_THREADS=$T numactl --interleave=all $BIN -n $N -nz $NZ -m $MODE -c $K 2>&1 | \
            tee $OUTDIR/n500k_T${T}_K${K}_run${RUN}.txt
    done
done
