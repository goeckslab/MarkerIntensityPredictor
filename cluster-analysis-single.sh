#!/bin/bash


source ./venv/bin/activate

folder=$1
subfolder=$2
runs=$3


for ((i = 1; i <= $runs; i++)); 
do
    folderName=Run$i

    echo Creating folder: $folderName
    
    mkdir -p results/cluster_analysis/$folderName


    python3 src/main.py cl -f \
    $folder/vae/$subfolder/vae_encoded_data.csv \
    $folder/pca/pca_encoded_data.csv  \
    $folder/vae/$subfolder/test_data.csv \
    -n AE PCA Raw

    for file in results/cluster_analysis/*; do 
        if [ -f "$file" ]; then 
            mv $file results/cluster_analysis/$folderName
        fi 
    done


done



