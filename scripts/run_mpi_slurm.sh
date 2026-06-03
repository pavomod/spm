#!/bin/bash
#SBATCH --job-name=spm_mpi
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH --output=%j_mpi.out

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
cd ~/gmodica5
OUTDIR=~/gmodica5/results/mpi_omp
mkdir -p $OUTDIR

for RUN in 1 2 3; do
    mpirun -np $SLURM_NTASKS ~/gmodica5/mpi_omp/mpi_omp -n 2000000 -nz 80000000 -m irregular 2>&1 | \
        tee $OUTDIR/strong_n2M_nodes${SLURM_NNODES}_p${SLURM_NTASKS}_t${OMP_NUM_THREADS}_run${RUN}.txt
done
