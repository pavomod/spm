#!/bin/bash
BIN=~/gmodica5/mpi_omp/mpi_omp
MODE=irregular
T=4
OUTDIR=~/gmodica5/results/mpi_omp/weak
mkdir -p $OUTDIR

# (NODES, NTASKS_PER_NODE, N, NZ)
# N and NZ scale linearly with NP = NODES × NTASKS_PER_NODE
configs=(
    "1 1  500000    20000000"    # NP=1
    "1 2  1000000   40000000"    # NP=2
    "1 4  2000000   80000000"    # NP=4,  1 node
    "2 4  4000000   160000000"   # NP=8,  2 nodes
    "4 4  8000000   320000000"   # NP=16, 4 nodes
    "8 4  16000000  640000000"   # NP=32, 8 nodes
)

for cfg in "${configs[@]}"; do
    read -r NODES NTPN N NZ <<< "$cfg"
    NP=$(( NODES * NTPN ))
    TAG="weak_nodes${NODES}_np${NP}_t${T}"

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
    echo "Submitted: $TAG  (NP=$NP, nodes=$NODES, T=$T, n=$N, nz=$NZ)"
done
