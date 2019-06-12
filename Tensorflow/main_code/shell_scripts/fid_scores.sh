#!/bin/bash
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem=16G
#SBATCH -p short
#SBATCH -o fid.out

sacct --format="CPUTime,MaxRSS"
python ../fid_computation/fid_scores.py
