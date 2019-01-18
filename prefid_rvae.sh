#!/bin/bash
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem=16G
#SBATCH -p short
#SBATCH -o prefid_rvae.out

sacct --format="CPUTime,MaxRSS"
python prefid_rvae.py $1 $2