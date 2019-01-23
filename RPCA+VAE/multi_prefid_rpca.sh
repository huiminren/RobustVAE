#!/bin/bash
for noise in $(seq 0.31 0.02 0.52)
do
    echo "noise" = $noise
    sbatch prefid_rpca.sh $noise
done
