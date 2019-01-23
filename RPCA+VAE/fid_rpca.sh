#!/bin/bash
#SBATCH -N 1
#SBATCH -n 16
#SBATCH --mem=64G
#SBATCH -p short
#SBATCH -o fid.out

sacct --format="CPUTime,MaxRSS"
python fid_scores.py
