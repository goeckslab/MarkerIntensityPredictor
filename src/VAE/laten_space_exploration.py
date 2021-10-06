import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tensorflow import keras
import pandas as pd
import argparse
import numpy as np
import umap

sns.set_theme()


# Weights
# https://stackoverflow.com/questions/58364974/how-to-load-trained-autoencoder-weights-for-decoder

# Latent space exploration
# https://towardsdatascience.com/variational-autoencoders-as-generative-models-with-keras-e0c79415a7eb
# https://towardsdatascience.com/intuitively-understanding-variational-autoencoders-1bfe67eb5daf

def explore_latent_space(arguments):
    """
    Explore the latent space
    """

    latent_space_data = pd.read_csv(arguments.file)
    markers = pd.read_csv(arguments.markers)
    model = keras.models.load_model(Path("results", "vae", "Run_0", "model"))

    print(model.encoder.summary())
    input()

    x_values = np.linspace(latent_space_data.min(), latent_space_data.max(), 100)
    count: int = 0
    generated_cells = pd.DataFrame()
    for ix, x in enumerate(x_values):
        # Extract first dimension of latent space
        first_dim = (x[[0][0]])
        # Fix remaining dimension on the mean
        mean = np.mean(x[[1, 2, 3, 4]])

        # Create new latent point with the extract and fixed dimensions
        latent_point = np.array([first_dim, mean, mean, mean, mean])

        # input()
        latent_point = latent_point.reshape(1, latent_point.shape[0])
        print(latent_point)
        # Generate new cell

        generated_cell = model.decoder.predict(latent_point)

        generated_cell = pd.Series(generated_cell[0])
        generated_cells = generated_cells.append(generated_cell, ignore_index=True)

        count += 1

    print(f"Generated {count} new cells.")

    generated_cells.columns = markers['names']
    generated_cells.to_csv(Path("results", "vae", "Run_0", "generated_cells.csv"), index=False)
    print(generated_cells)

    difference = generated_cells.diff(axis=0)
    print(difference)
    input()


    plt.figure(figsize=(20, 9))
    ax = sns.heatmap(generated_cells, vmin=generated_cells.min().min(), vmax=generated_cells.max().max())
    fig = ax.get_figure()
    plt.xlabel("Marker")
    plt.ylabel("Cell")
    # ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=90)
    plt.tight_layout()
    fig.savefig(Path(f"results", "vae", "Run_0", "heatmap.png"))

    plt.close('all')

    fit = umap.UMAP()
    mapping = fit.fit_transform(generated_cells)

    plot = sns.scatterplot(data=mapping, x=mapping[:, 0], y=mapping[:, 1], hue=pd.Series(generated_cells.index))
    fig = plot.get_figure()
    fig.savefig(Path(f"results", "vae", "Run_0", "cells.png"))
    plt.close('all')


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", type=str, required=True, action="store",
                        help="The file to load and use")
    parser.add_argument("-m", "--markers", type=str, required=True, action="store",
                        help="The markers file to load and use")

    return parser.parse_args()


if __name__ == "__main__":
    args = args_parser()
    explore_latent_space(args)
