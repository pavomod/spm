#!/bin/bash
BIN=~/gmodica5/omp_ws/omp_ws
N=500000
NZ=20000000
MODE=irregular
OUTDIR=~/gmodica5/results/omp_ws
mkdir -p $OUTDIR

for T in 1 2 4 8 16 32; do
    echo "=== T=$T ==="
    for RUN in 1 2 3; do
        OMP_PROC_BIND=close OMP_PLACES=cores OMP_NUM_THREADS=$T numactl --interleave=all $BIN -n $N -nz $NZ -m $MODE 2>&1 | \
            tee $OUTDIR/n500k_T${T}_run${RUN}.txt
    done
done
