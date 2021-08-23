#!/bin/bash


source ./venv/bin/activate

folder=$1
runs=$2


for ((i = 1; i <= $runs; i++)); 
do
    folderName=Run$i

    echo Creating folder: $folderName
    
    mkdir -p results/cluster_analysis/$folderName


    python3 src/main.py cl -f \
    $folder/vae/9-2-1/vae_encoded_data.csv \
    $folder/pca/9-2-1/pca_encoded_data.csv  \
    $folder/vae/9-2-1/test_data.csv \
    $folder/vae/9-2-2/vae_encoded_data.csv \
    $folder/pca/9-2-2/pca_encoded_data.csv \
    $folder/vae/9-2-2/test_data.csv \
    $folder/vae/9-3-1/vae_encoded_data.csv \
    $folder/pca/9-3-1/pca_encoded_data.csv \
    $folder/vae/9-3-1/test_data.csv \
    $folder/vae/9-3-2/vae_encoded_data.csv \
    $folder/pca/9-3-2/pca_encoded_data.csv \
    $folder/vae/9-3-2/test_data.csv \
    -n vae pca non

    for file in results/cluster_analysis/*; do 
        if [ -f "$file" ]; then 
            mv $file results/cluster_analysis/$folderName
        fi 
    done


done



