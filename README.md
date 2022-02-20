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

## Deep learning model training

### Setup & Configuration

1. Create a virtual environment
2. `pip install -r requirements.txt` will install all required packages

### ML Flow Integration

[MLFlow](https://www.mlflow.org) is being used for experiment tracking.  
Although a results folder is being used locally, all data is being stored using the backend of MLFlow.  
**Do NOT rely on the local results folder, it is only temporary.**

### Usage

`python VAE/main.py model --file data/HTA9-3_Bx2_HMS_Tumor_quant.csv`

#### Available Arguments

| Long           | Short | Description                                                                                                                                          | Required |   
|----------------|-------|------------------------------------------------------------------------------------------------------------------------------------------------------|----------|
| --file         | x     | The file to use for training the models                                                                                                              | ✓        |
| --run          | -r    | The name of the run                                                                                                                                  | ✓        |
| --experiment   | -e    | The name of the associated experiment. <br/>If no experiment with this name exists and new one will be created. <br/> Default: Default experiment    |          |
| --morph        | x     | Should morphological features be included for training and evaluation? Default: true                                                                 |          |
| --mode         | x     | Possible values: none, ae, vae. If ae or vae is selected only the respective model is being trained. No comparison will be done. <br/>Default: none. |          |
| --tracking_url | -t    | The tracking url for the mlflow tracking server. Points to localhost as default. Please specify a valid url. <br/> Example: http:127.0.0.1:5000      |          |




## Latent space exploration

A tool to explore the latent space of a VAE is included.   
The tool is using Streamlit and MLFlow to provide interactive as well as tracking functionalities.