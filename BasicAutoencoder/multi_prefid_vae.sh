#!/bin/bash
for noise in $(seq 0.01 0.04 0.51)
do
    echo "noise" = $noise
    sbatch prefid_vae.sh $noise
done
