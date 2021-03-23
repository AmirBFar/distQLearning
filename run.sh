#!/bin/bash
#SBATCH --job-name=RL
#SBATCH --ntasks=50
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=output.out
#SBATCH --error=error.err
#SBATCH --export=ALL

#srun --mpi=pmi2  python3 age_mpi_run.py 
srun --mpi=pmi2  python3 age_mpi_plot.py
