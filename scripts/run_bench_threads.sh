#!/bin/bash
BIN=~/gmodica5/threads/threads
N=500000
NZ=20000000
MODE=irregular
OUTDIR=~/gmodica5/results/threads
mkdir -p $OUTDIR

for T in 1 2 4 8 16 32; do
    echo "=== T=$T ==="
    for RUN in 1 2 3; do
        numactl --interleave=all $BIN -n $N -nz $NZ -m $MODE -t $T 2>&1 | tee $OUTDIR/n500k_T${T}_run${RUN}.txt
    done
done
