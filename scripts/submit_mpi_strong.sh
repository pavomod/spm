#!/bin/bash
BIN=~/gmodica5/mpi_omp/mpi_omp
N=2000000
NZ=80000000
MODE=irregular
T=4
OUTDIR=~/gmodica5/results/mpi_omp/strong
mkdir -p $OUTDIR

# (NODES, NTASKS_PER_NODE)  →  NP = NODES × NTASKS_PER_NODE
# cpus-per-task = T (OpenMP threads per rank)
# Constraint: NTASKS_PER_NODE × T ≤ 16 (physical cores per node)
configs=(
    "1 1"    # NP=1,  1 node   → 4  total threads
    "1 2"    # NP=2,  1 node   → 8  total threads
    "1 4"    # NP=4,  1 node   → 16 total threads (1 full node)
    "2 4"    # NP=8,  2 nodes  → 32 total threads
    "4 4"    # NP=16, 4 nodes  → 64 total threads
    "8 4"    # NP=32, 8 nodes  → 128 total threads
)

for cfg in "${configs[@]}"; do
    read -r NODES NTPN <<< "$cfg"
    NP=$(( NODES * NTPN ))
    TAG="strong_n2M_nodes${NODES}_np${NP}_t${T}"

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
    echo "Submitted: $TAG  (NP=$NP, nodes=$NODES, T=$T)"
done
