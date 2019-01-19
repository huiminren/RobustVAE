#!/bin/bash
for lambda in 1 5 10 15 20 25 50 70 100 250
do
    for noise in $(seq 0.01 0.04 0.51);
    do
        echo "lambda" = $lambda, "noise" = $noise
        sbatch prefid_rvae.sh $lambda $noise
    done
done