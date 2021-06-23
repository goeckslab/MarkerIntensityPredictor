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

### Linear mode

1. Create or reuse your venv.
2. ```./start_linear.sh```
3. Use arguments to start different modes


### Auto Encoders

#### Vanilla Auto Encoder

1. Create or reuse your venv.
2. ```./start_ae.sh```

#### Denoising Auto Encoder

1. Create or reuse your venv.
2. ```./start_dae.sh```


### Plotting

1. Create or reuse your venv.