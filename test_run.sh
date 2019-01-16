#!/bin/bash
#SBATCH -N 1
#SBATCH -n 16
#SBATCH --mem=64G
#SBATCH -p short
#SBATCH -o GAN_lambda_$1_noise_$2.out
#SBATCH --job-name=lambda_$1_noise_$2.out

sacct --format="CPUTime,MaxRSS"
python test.py $1 $2
