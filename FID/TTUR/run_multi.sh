#!/bin/bash

for noise in $(seq 0.0 0.1 1)
do
	for lambda in 1 5 10 15 20 25 50 70 100:
	do
		sed "41c l=$lambda" test.py > "test_$lambda_$noise.py"
        sed -i "42c nr=$noise" "test_$lambda_$noise.py"
        sed "6c #SBATCH -o FID_$lambda_$noise.out" test_run.sh > "test_run_$lambda_$noise.sh"
        sed -i "7c #SBATCH --job-name='job_$lambda_$noise'" "test_run_$lambda_$noise.sh"
        sed -i "10c python test_$lambda_$noise.py" "test_run_$lambda_$noise.sh"
        sbatch "test_run_$lambda_$noise.sh"
	done
done
