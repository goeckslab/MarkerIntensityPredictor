import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import streamlit as st

class Data:
    init_inputs: np.array
    init_markers: pd.DataFrame()
    init_X_train = pd.DataFrame()
    init_X_test = pd.DataFrame()
    init_X_val = pd.DataFrame()

    inputs: np.array
    markers = pd.DataFrame()
    X_train_noise = pd.DataFrame()
    X_train: pd.DataFrame()
    X_test: pd.DataFrame()
    X_val: pd.DataFrame()

    def __init__(self, normalize, inputs=None, train: np.array = None, test: np.array = None, markers=None):
        # Train and Test data splits were provided
        if train is not None and test is not None:
            self.init_markers = markers
            # Split train set into train and validation set
            X_train, X_val = train_test_split(train, test_size=0.05, random_state=1, shuffle=True)

            self.init_X_train = X_train
            self.init_X_test = test
            self.init_X_val = X_val

            # Store the normalized data
            self.markers = self.init_markers
            self.X_train = normalize(X_train.copy())
            self.X_test = normalize(test.copy())
            self.X_val = normalize(X_val.copy())
            self.inputs_dim = train.shape[1]

        # Do a normal split on the given input data set
        else:
            self.init_inputs = inputs
            self.init_markers = markers
            X_dev, X_val = train_test_split(self.init_inputs, test_size=0.05, random_state=1, shuffle=True)
            X_train, X_test = train_test_split(X_dev, test_size=0.25, random_state=1, shuffle=True)

            self.init_X_train = X_train
            self.init_X_test = X_test
            self.init_X_val = X_val

            # Store the normalized data
            self.markers = self.init_markers

            self.inputs = normalize(np.array(inputs))
            self.X_train = normalize(X_train)
            self.X_test = normalize(X_test)
            self.X_val = normalize(X_val)
            self.inputs_dim = self.inputs.shape[1]
