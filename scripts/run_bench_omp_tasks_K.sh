#!/bin/bash
BIN=~/gmodica5/omp_tasks/omp_tasks
N=500000
NZ=20000000
MODE=irregular
T=16
OUTDIR=~/gmodica5/results/omp_tasks
mkdir -p $OUTDIR

for K in 16 64 256 1024 4096; do
    echo "=== T=$T, K=$K ==="
    for RUN in 1 2 3; do
        OMP_PROC_BIND=close OMP_PLACES=cores OMP_NUM_THREADS=$T numactl --interleave=all $BIN -n $N -nz $NZ -m $MODE -c $K 2>&1 | \
            tee $OUTDIR/n500k_T${T}_K${K}_run${RUN}.txt
    done
done
