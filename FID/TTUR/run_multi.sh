#!/bin/bash

list1="1 5 10 15 20 25 50 70 100"

list1_x=($list1)

length=${#list1_x[@]} 

for noise in $(seq 0.0 0.1 1)
do 
    for ((i=0; i<${length}; i++));
    do
        sed "41c l=${list1_x[$i]}" test.py > "test_${list1_x[$i]}_$noise.py"
        sed -i "42c nr=$noise" "test_${list1_x[$i]}_$noise.py"
        sed "6c #SBATCH -o FID_${list1_x[$i]}_$noise.out" test_run.sh > "test_run_${list1_x[$i]}_$noise.sh"
        sed -i "7c #SBATCH --job-name='job_${list1_x[$i]}_$noise'" "test_run_${list1_x[$i]}_$noise.sh"
        sed -i "10c python test_${list1_x[$i]}_$noise.py" "test_run_${list1_x[$i]}_$noise.sh"
        sbatch "test_run_${list1_x[$i]}_$noise.sh"
    done
done


