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

## Variational Auto Encoder

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

| Param  | Description                                      |     |     |     |
|--------|--------------------------------------------------|-----|-----|-----|
| --file | The file the tool should use to train all models |     |     |     |
|        |                                                  |     |     |     |
|        |                                                  |     |     |     |
