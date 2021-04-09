import numpy as np
import pandas as pd
import keras
import re
import matplotlib.pyplot as plt
from random import randrange
from keras import layers, regularizers
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from shared.services.plots import Plots
from VAE.statistics import Stats
from VAE.entities.sampling import Sampling
from VAE.entities.vae import VAE


def check_data(inputs):
    rnd = randrange(0, inputs.shape[1])
    # Mean should be zero and standard deviation
    # should be 1. However, due to some challenges
    # relationg to floating point positions and rounding,
    # the values should be very close to these numbers.
    # For details, see:
    # https://stackoverflow.com/a/40405912/947889
    # Hence, we assert the rounded values.
    print(inputs[:, rnd].std())
    print(inputs[:, rnd].mean())


def normalize(data):
    # Input data contains some zeros which results in NaN (or Inf)
    # values when their log10 is computed. NaN (or Inf) are problematic
    # values for downstream analysis. Therefore, zeros are replaced by
    # a small value; see the following thread for related discussion.
    # https://www.researchgate.net/post/Log_transformation_of_values_that_include_0_zero_for_statistical_analyses2
    data[data == 0] = 1e-32
    data = np.log10(data)

    standard_scaler = StandardScaler()
    data = standard_scaler.fit_transform(data)
    data = data.clip(min=-10, max=10)

    min_max_scaler = MinMaxScaler(feature_range=(-1, 1))
    data = min_max_scaler.fit_transform(data)
    return data


def get_data():
    # Load Data
    # We load data into RAM since data is small and will fit in memory.
    cells = pd.read_csv("tumor_cycif_v2.csv", header=0)

    # Keeps only the 'interesting' columns.
    cells = cells.filter(regex="nucleiMasks$", axis=1).filter(regex="^(?!(DAPI|AF))", axis=1)
    markers = cells.columns
    markers = [re.sub("_nucleiMasks", "", x) for x in markers]

    return cells, markers


if __name__ == "__main__":
    inputs, markers = get_data()
    inputs = np.array(inputs)
    # random_state is a random seed to have a reproducible shuffling (only for dev purpose).
    X_dev, X_val = train_test_split(inputs, test_size=0.05, random_state=1, shuffle=True)
    X_train, X_test = train_test_split(X_dev, test_size=0.25, random_state=1)

    # This is primarily for the purpose of
    # being able to see how much data is removed
    # (as part of outlier removal) and plot
    # the changes in data distribution.
    init_inputs = inputs
    init_X_train = X_train
    init_X_test = X_test
    init_X_val = X_val

    inputs = normalize(inputs)
    X_train = normalize(X_train)
    X_test = normalize(X_test)
    X_val = normalize(X_val)

    # NOTE: the number of cells in X_train, X_test, and X_val would NOT
    # add up to the number of cells in inputs if outliers are removed.
    # It is because outliers are removed after the cells in inputs
    # is split between train, test, and val sets.

    Stats.print_data_overview(init_inputs, inputs, X_train, init_X_train, init_X_test, init_X_val,X_test, X_val)
    check_data(inputs)

    fig, axs = plt.subplots(nrows=4, ncols=2, figsize=(12, 9), dpi=300)
    plt.subplots_adjust(wspace=0.20, hspace=0.50)
    Plots.plot_dists(axs[0, 0], init_inputs, "Input")
    Plots.plot_dists(axs[0, 1], inputs, "Normalized Input")
    Plots.plot_dists(axs[1, 0], init_X_train, "X-Train")
    Plots.plot_dists(axs[1, 1], X_train, "Normalized X-Train")
    Plots.plot_dists(axs[2, 0], init_X_test, "X-Test")
    Plots.plot_dists(axs[2, 1], X_test, "Normalized X-Test")
    Plots.plot_dists(axs[3, 0], init_X_val, "X-Validation")
    Plots.plot_dists(axs[3, 1], X_val, "Normalized X-Validation")
    plt.show()

    # Build the encoder
    # length of latent vector.
    latent_dim = 6

    inputs_dim = inputs.shape[1]
    r = regularizers.l1(10e-5)
    activation = "linear"

    encoder_inputs = keras.Input(shape=(inputs_dim))
    h1 = layers.Dense(inputs_dim, activation=activation, activity_regularizer=r)(encoder_inputs)
    h2 = layers.Dense(inputs_dim / 2, activation=activation, activity_regularizer=r)(h1)
    h3 = layers.Dense(inputs_dim / 3, activation=activation, activity_regularizer=r)(h2)

    # The following variables are for the convenience of building the decoder.
    # last layer before flatten
    lbf = h3
    # shape before flatten.
    sbf = keras.backend.int_shape(lbf)[1:]
    # neurons count before latent dim
    nbl = np.prod(sbf)

    z_mean = layers.Dense(latent_dim, name="z_mean")(lbf)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(lbf)
    z = Sampling()([z_mean, z_log_var])
    encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
    encoder.summary()

    # Build the decoder
    decoder_inputs = keras.Input(shape=(latent_dim,))
    h1 = layers.Dense(nbl, activation=activation)(decoder_inputs)
    h2 = layers.Dense(inputs_dim / 2, activation=activation)(h1)

    decoder_outputs = layers.Dense(inputs_dim)(h2)
    decoder = keras.Model(decoder_inputs, decoder_outputs, name="decoder")
    decoder.summary()

    # Visualize the model.
    # tf.keras.utils.plot_model(model, to_file="model.png")

    # Train the VAE
    # Create the VAR, compile, and run.
    vae = VAE(encoder, decoder)
    vae.compile(optimizer=keras.optimizers.Adam(lr=0.0005))
    history = vae.fit(X_train, validation_data=(X_test, X_test), epochs=100, batch_size=32, shuffle=True, verbose=0)

    # Make some predictions
    cell = X_val[0]
    cell = cell.reshape(1, cell.shape[0])
    mean, log_var, z = encoder.predict(cell)
    encoded_cell = z
    decoded_cell = decoder.predict(encoded_cell)
    var_cell = vae.predict(cell)
    print(f"Input shape:\t{cell.shape}")
    print(f"Encoded shape:\t{encoded_cell.shape}")
    print(f"Decoded shape:\t{decoded_cell.shape}")
    print(f"\nInput:\n{cell[0]}")
    print(f"\nEncoded:\n{encoded_cell[0]}")
    print(f"\nDecoded:\n{decoded_cell[0]}")

    # Explore the latent space
    # Get all the possible permutations. For instance:
    # Input with three possible values for two latent
    # variables:
    # [[0.01 0.5  0.99], [0.01 0.5  0.99]]
    #
    # All the possible combinations:
    # [[[0.01 0.01], [0.5  0.01], [0.99 0.01]],
    #  [[0.01 0.5 ], [0.5  0.5 ], [0.99 0.5 ]],
    #  [[0.01 0.99], [0.5  0.99], [0.99 0.99]]]

    # Linearly spaced latent variables.
    # Values are transformed through the inverse CDF (PPF)
    # of the Gaussian since the prior of the latent space
    # is Gaussian.
    step_size = 4
    z = np.array([np.linspace(-4, 4, step_size)] * latent_dim)
    # z = norm.ppf(z)

    z_grid = np.dstack(np.meshgrid(*z))
    z_grid = z_grid.reshape(step_size ** latent_dim, latent_dim)

    x_pred_grid = decoder.predict(z_grid)
    x_pred_grid = np.block(list(map(list, x_pred_grid)))

    Plots.plot_distribution_of_latent_variables(encoder, X_train, latent_dim, step_size, z)
    Plots.plot_model_performance(history)
    Plots.plot_markers(X_train, X_test, X_val, markers)
    Plots.plot_reconstructed_markers(z_grid, x_pred_grid, markers)
    Plots.latent_space_cluster(X_train, vae)
    Plots.plot_reconstructed_intensities(vae, X_val, markers)
