#!/bin/bash


source ./venv/bin/activate

runs=$1



for ((i = 1; i <= $runs; i++)); 
do
    folderName=Run$i

    echo Creating folder: $folderName

    mkdir -p results/cluster_analysis/$folderName


    python3 src/main.py cl -f \
    /Users/raphael/Box/Goecks\ Precision\ Oncology\ Analytics/Projects/Raphael/Marker\ Intensity\ Predictions/Experiment\ 07-13-21/VAE/9-2-1/encoded_data.csv \
    /Users/raphael/Box/Goecks\ Precision\ Oncology\ Analytics/Projects/Raphael/Marker\ Intensity\ Predictions/Experiment\ 07-13-21/PCA/9-2-1/encoded_data.csv  \
    /Users/raphael/Box/Goecks\ Precision\ Oncology\ Analytics/Projects/Raphael/Marker\ Intensity\ Predictions/Experiment\ 07-13-21/VAE/9-2-1/test_data.csv \
    /Users/raphael/Box/Goecks\ Precision\ Oncology\ Analytics/Projects/Raphael/Marker\ Intensity\ Predictions/Experiment\ 07-13-21/VAE/9-2-2/encoded_data.csv \
    /Users/raphael/Box/Goecks\ Precision\ Oncology\ Analytics/Projects/Raphael/Marker\ Intensity\ Predictions/Experiment\ 07-13-21/PCA/9-2-2/encoded_data.csv \
    /Users/raphael/Box/Goecks\ Precision\ Oncology\ Analytics/Projects/Raphael/Marker\ Intensity\ Predictions/Experiment\ 07-13-21/VAE/9-2-2/test_data.csv \
    /Users/raphael/Box/Goecks\ Precision\ Oncology\ Analytics/Projects/Raphael/Marker\ Intensity\ Predictions/Experiment\ 07-13-21/VAE/9-3-1/encoded_data.csv \
    /Users/raphael/Box/Goecks\ Precision\ Oncology\ Analytics/Projects/Raphael/Marker\ Intensity\ Predictions/Experiment\ 07-13-21/PCA/9-3-1/encoded_data.csv \
    /Users/raphael/Box/Goecks\ Precision\ Oncology\ Analytics/Projects/Raphael/Marker\ Intensity\ Predictions/Experiment\ 07-13-21/VAE/9-3-1/test_data.csv \
    /Users/raphael/Box/Goecks\ Precision\ Oncology\ Analytics/Projects/Raphael/Marker\ Intensity\ Predictions/Experiment\ 07-13-21/VAE/9-3-2/encoded_data.csv \
    /Users/raphael/Box/Goecks\ Precision\ Oncology\ Analytics/Projects/Raphael/Marker\ Intensity\ Predictions/Experiment\ 07-13-21/PCA/9-3-2/encoded_data.csv \
    /Users/raphael/Box/Goecks\ Precision\ Oncology\ Analytics/Projects/Raphael/Marker\ Intensity\ Predictions/Experiment\ 07-13-21/VAE/9-3-2/test_data.csv \
    -n vae pca non

    for file in results/cluster_analysis/*; do 
        if [ -f "$file" ]; then 
            mv $file results/cluster_analysis/$folderName
        fi 
    done


done



