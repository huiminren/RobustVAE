#!/bin/bash
for noise in $(seq 0.01 0.02 0.51)
do
    echo "noise" = $noise
    sbatch --job-name=vae_$noise --output=vae_n_$noise.out vae.sh $noise
done
