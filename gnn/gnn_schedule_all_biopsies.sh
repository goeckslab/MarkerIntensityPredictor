
#!/bin/bash

iterations=$1
repetitions=$2
biopsies=('9_2_1' '9_2_2' '9_3_1' '9_3_2' '9_14_1' '9_14_2' '9_15_1' '9_15_2')
spatial_radius=(23 46 92 138 184)


echo "iterations=${iterations}"

for biopsy in "${biopsies[@]}"; do
  for radius in "${spatial_radius[@]}"; do
    for i in $(seq 1 $iterations)
      do
        #echo "${biopsy}" "${radius}"
        sbatch ./gnn/gnn.sh "${biopsy}" "mean" "ip" "${radius}" "${repetitions}"
        sbatch ./gnn/gnn.sh "${biopsy}" "mean" "exp" "${radius}" "${repetitions}"
        sbatch ./gnn/gnn.sh "${biopsy}" "zero" "ip" "${radius}" "${repetitions}"
        sbatch ./gnn/gnn.sh "${biopsy}" "zero" "exp" "${radius}" "${repetitions}"
      done
    done
done
