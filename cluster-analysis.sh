#!/bin/bash


source ./venv/bin/activate

runs=$1


for ((i = 1; i <= $runs; i++)); 
do
    echo $i
    folderName=Run$i

    echo $folderName

    mkdir -p results/cluster_analysis/$folderName


    python3 src/main.py cl -f \
    /Users/raphael/Box/Goecks\ Precision\ Oncology\ Analytics/Projects/Raphael/Marker\ Intensity\ Predictions/Experiment\ 07-13-21/Cluster\ analysis/9-2-1/encoded_data_pg_clusters.csv \
    /Users/raphael/Box/Goecks\ Precision\ Oncology\ Analytics/Projects/Raphael/Marker\ Intensity\ Predictions/Experiment\ 07-13-21/Cluster\ analysis/9-2-1/pca_pg_clusters.csv  \
    /Users/raphael/Box/Goecks\ Precision\ Oncology\ Analytics/Projects/Raphael/Marker\ Intensity\ Predictions/Experiment\ 07-13-21/Cluster\ analysis/9-2-1/test_data_pg_clusters.csv \
    /Users/raphael/Box/Goecks\ Precision\ Oncology\ Analytics/Projects/Raphael/Marker\ Intensity\ Predictions/Experiment\ 07-13-21/Cluster\ analysis/9-2-2/encoded_data_pg_clusters.csv \
    /Users/raphael/Box/Goecks\ Precision\ Oncology\ Analytics/Projects/Raphael/Marker\ Intensity\ Predictions/Experiment\ 07-13-21/Cluster\ analysis/9-2-2/pca_pg_clusters.csv \
    /Users/raphael/Box/Goecks\ Precision\ Oncology\ Analytics/Projects/Raphael/Marker\ Intensity\ Predictions/Experiment\ 07-13-21/Cluster\ analysis/9-2-2/test_data_pg_clusters.csv \
    /Users/raphael/Box/Goecks\ Precision\ Oncology\ Analytics/Projects/Raphael/Marker\ Intensity\ Predictions/Experiment\ 07-13-21/Cluster\ analysis/9-3-1/encoded_data_pg_clusters.csv \
    /Users/raphael/Box/Goecks\ Precision\ Oncology\ Analytics/Projects/Raphael/Marker\ Intensity\ Predictions/Experiment\ 07-13-21/Cluster\ analysis/9-3-1/pca_pg_clusters.csv \
    /Users/raphael/Box/Goecks\ Precision\ Oncology\ Analytics/Projects/Raphael/Marker\ Intensity\ Predictions/Experiment\ 07-13-21/Cluster\ analysis/9-3-1/test_data_pg_clusters.csv \
    /Users/raphael/Box/Goecks\ Precision\ Oncology\ Analytics/Projects/Raphael/Marker\ Intensity\ Predictions/Experiment\ 07-13-21/Cluster\ analysis/9-3-2/encoded_data_pg_clusters.csv \
    /Users/raphael/Box/Goecks\ Precision\ Oncology\ Analytics/Projects/Raphael/Marker\ Intensity\ Predictions/Experiment\ 07-13-21/Cluster\ analysis/9-3-2/pca_pg_clusters.csv \
    /Users/raphael/Box/Goecks\ Precision\ Oncology\ Analytics/Projects/Raphael/Marker\ Intensity\ Predictions/Experiment\ 07-13-21/Cluster\ analysis/9-3-2/test_data_pg_clusters.csv \
    -n vae pca non

    for file in results/cluster_analysis/*; do 
        if [ -f "$file" ]; then 
            mv $file results/cluster_analysis/$folderName/$file
        fi 
    done


done



