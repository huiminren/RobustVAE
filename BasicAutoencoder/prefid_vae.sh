#!/bin/bash
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem=16G
#SBATCH -p short
#SBATCH -o pfvae_n_$1.out
#SBATCH --job-name=pfvae_$1.out

sacct --format="CPUTime,MaxRSS"
python prefid_vae.py $1
