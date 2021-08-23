# Marker Intensity Predictor

This repo contains different part of Marker Intensity Research, based on HTAN data files.

It includes:
1. Jupyter notebooks
2. A linear regression model to calculate LR baseline data
3. Different auto encoders to train models 
4. A plotting package to combine the results of linear regression and different auto encoders



## Jupyter notebooks
Run the following to start the jupyter notebook:

```shell script
$ virtualenv .venv && source .venv/bin/activate && python -m pip install -r requirements.txt && python -m ipykernel install --user --name=.venv
$ jupyter labextension install jupyterlab-plotly@4.14.2
$ jupyter-lab
```


## Scripts


#### VAE

Example: 
```python3 src/main.py vae -f data/HTA9-2_Bx1_HMS_Tumor_quant.csv ```

Will perform the following tasks.
1. Trains a model using a setup vae
2. Decode the data
3. Encode the data
4. Writes the resulting dataset to disk


#### PCA

Example: 
```python3 src/main.py pca -f data/HTA9-2_Bx1_HMS_Tumor_quant.csv ```

Will perform the following tasks.
1. Trains a model using pca
2. Decode the data
3. Encode the data
4. Writes the resulting dataset to disk




### Cluster Analysis

Example: ```python3 src/main.py cl -f results/vae/vae_encoded_data.csv results/vae/test_data.csv results/pca/pca_encoded_data.csv -n vae non pca ```

Will perform a cluster analysis using kmeans and phenograph.



#### Clustering


Supported arguments:

| Short   |  Long  |  Description  | Multiple args  |
|---|---|---|---|
| -f  | --file  | Specifies the files to use for the algorithm  | x  |
| -n | --names  | Specify the names of the groups associated to the files  | x  |



