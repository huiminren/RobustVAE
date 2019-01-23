#!/bin/bash
for noise in $(seq 0.31 0.02 0.51)
do
    echo "noise" = $noise
    sbatch --job-name=rpca_$noise --output=rpca_n_$noise.out rpca.sh $noise
done
