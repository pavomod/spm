#!/bin/bash
# NP×T interaction grid — run from FRONTEND, submits via sbatch.
# Fixed problem: n=2M, nz=80M, irregular.
# Compute nodes: 16 physical cores/node (8 cores × 2 sockets, HT → 32 logical).
# Constraint: NTPN × T ≤ 16, NODES ≤ 8.
# Format: "NODES  NTPN  T"  →  NP = NODES × NTPN

BIN=~/gmodica5/mpi_omp/mpi_omp
N=2000000
NZ=80000000
MODE=irregular
OUTDIR=~/gmodica5/results/mpi_omp/npt_grid
mkdir -p "$OUTDIR"

configs=(
    # 1 node (16 physical cores) — NP×T tradeoff, fixed total resources
    "1   1  16"    # NP=1,  T=16  pure OpenMP
    "1   2   8"    # NP=2,  T=8
    "1   4   4"    # NP=4,  T=4
    "1   8   2"    # NP=8,  T=2
    "1  16   1"    # NP=16, T=1   pure MPI intra-node

    # 2 nodes (32 physical cores)
    "2   1  16"    # NP=2,  T=16
    "2   2   8"    # NP=4,  T=8
    "2   4   4"    # NP=8,  T=4
    "2   8   2"    # NP=16, T=2
    "2  16   1"    # NP=32, T=1

    # 4 nodes (64 physical cores)
    "4   1  16"    # NP=4,  T=16
    "4   2   8"    # NP=8,  T=8
    "4   4   4"    # NP=16, T=4
    "4   8   2"    # NP=32, T=2
    "4  16   1"    # NP=64, T=1

    # 8 nodes (128 physical cores)
    "8   1  16"    # NP=8,  T=16
    "8   2   8"    # NP=16, T=8
    "8   4   4"    # NP=32, T=4
    "8   8   2"    # NP=64, T=2
    "8  16   1"    # NP=128, T=1
)

for cfg in "${configs[@]}"; do
    read -r NODES NTPN T <<< "$cfg"
    NP=$(( NODES * NTPN ))
    TOTAL=$(( NP * T ))
    TAG="npt_nodes${NODES}_np${NP}_t${T}"

    TMPSCRIPT=$(mktemp /tmp/spm_XXXXX.sh)
    cat > "$TMPSCRIPT" << EOF
#!/bin/bash
#SBATCH --nodes=$NODES
#SBATCH --ntasks-per-node=$NTPN
#SBATCH --cpus-per-task=$T
#SBATCH --job-name=spm_${TAG}
#SBATCH --output=${OUTDIR}/${TAG}_%j.slurm.out

export OMP_NUM_THREADS=$T
export OMP_PROC_BIND=close
export OMP_PLACES=cores
cd ~/gmodica5
for RUN in 1 2 3; do
    mpirun -np $NP $BIN -n $N -nz $NZ -m $MODE 2>&1 | tee ${OUTDIR}/${TAG}_run\${RUN}.txt
done
EOF
    sbatch "$TMPSCRIPT"
    rm "$TMPSCRIPT"
    echo "Submitted: $TAG  (NP=$NP, nodes=$NODES, T=$T, total_cores=$TOTAL)"
done
