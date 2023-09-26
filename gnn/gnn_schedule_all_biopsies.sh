
#!/bin/bash

experiments=$1
subsets=$2
biopsies=('9_2_1' '9_2_2' '9_3_1' '9_3_2' '9_14_1' '9_14_2' '9_15_1' '9_15_2')
spatial_radius=(23 46 92 138 184)

# if subsets is not set, set it to 1
if [ "$subsets" == "" ];
then
  echo "Subsets was not set. Setting it to 1."
  subsets=1
fi


for biopsy in "${biopsies[@]}"; do
  for radius in "${spatial_radius[@]}"; do
    for i in $(seq 1 $experiments)
      do
        echo biopsy="${biopsy}" radius="${radius}" subsets="${subsets}" experiments="${experiments}" i="${i}"
        sbatch ./gnn/gnn.sh "${biopsy}" "mean" "ip" "${radius}" "${subsets}"
        sbatch ./gnn/gnn.sh "${biopsy}" "mean" "exp" "${radius}" "${subsets}"
        sbatch ./gnn/gnn.sh "${biopsy}" "zero" "ip" "${radius}" "${subsets}"
        sbatch ./gnn/gnn.sh "${biopsy}" "zero" "exp" "${radius}" "${subsets}"
      done
    done
done
