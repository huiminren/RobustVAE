#!/bin/bash
for noise in $(seq 0.0 0.1 1)
do
    echo "noise = $noise"
    sbatch --job-name=GAN_$noise --output=GAN_noise_$noise.out test_run.sh $noise
done
