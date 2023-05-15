
#!/bin/bash

iterations=$1
biopsies=('9_2_1' '9_2_2' '9_3_1' '9_3_2' '9_14_1' '9_14_2' '9_15_1' '9_15_2')
spatial_radius=(23 46 92 138 184)

for biopsy in "${biopsies[@]}"; do
  for radius in "${spatial_radius[@]}"; do
    for i in $(seq 1 $iterations)
      do
        #echo "${biopsy}" "${radius}"
        sbatch ./gnn/gnn.sh "${data_path}" "mean" "ip" "${radius}"
        sbatch ./gnn/gnn.sh "${data_path}" "mean" "exp" "${radius}"
        sbatch ./gnn/gnn.sh "${data_path}" "zero" "ip" "${radius}"
        sbatch ./gnn/gnn.sh "${data_path}" "zero" "exp" "${radius}"
      done
    done
done
