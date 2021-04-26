# Marker Intensity Predictor

This repo contains different part of Marker Intensity Research, based on HTAN data files.

It includes:
1. Jupyter notebooks
2. A linear regression model to calculate LR baseline data
3. A Ludwig AI tool to create yaml files for Ludwig

Run the following to start the jupyter notebook:

```shell script
$ virtualenv .venv && source .venv/bin/activate && python -m pip install -r requirements.txt && python -m ipykernel install --user --name=.venv
$ jupyter labextension install jupyterlab-plotly@4.14.2
$ jupyter-lab
```
