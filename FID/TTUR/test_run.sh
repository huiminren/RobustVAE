#!/bin/bash
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem=64G
#SBATCH -p short
#SBATCH -o GAN_copy_no1024.out
#SBATCH --job-name="hello_test"

sacct --format="CPUTime,MaxRSS"
python GAN_copy_no1024.py
