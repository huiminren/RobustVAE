#!/bin/bash
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem=16G
#SBATCH -p short
#SBATCH -o fid_scores_rvae.out

sacct --format="CPUTime,MaxRSS"
python fid_scores.py $1 $2