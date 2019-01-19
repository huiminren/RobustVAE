#!/bin/bash
#SBATCH -N 1
#SBATCH -n 16
#SBATCH --mem=64G
#SBATCH -p short
#SBATCH -o rvae_sp.out
#SBATCH --job-name=rvae_sp

sacct --format="CPUTime,MaxRSS"
python RobustVariationalAutoencoder.py $1 $2