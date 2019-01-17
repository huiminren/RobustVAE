#!/bin/bash
for lambda in 1 5 10 15 20 25 50 70 100 250
do
    for noise in $(seq 0.01 0.02 0.29);
    do
        echo "lambda" = $lambda, "noise" = $noise
        sbatch rvae_sp.sh $lambda $noise
    done
done