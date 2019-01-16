#!/bin/bash
for noise in $(seq 0.1 0.1 1)
do
    for lambda in 0.1 1 5 10 15 20 25 50 70 100 250;
    do
        echo "lambda = $lambda, noise = $noise"
        sbatch test_run.sh $lambda $noise
    done
done
