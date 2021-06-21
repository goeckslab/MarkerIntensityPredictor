import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


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

    def __init__(self, inputs, markers, normalize):
        self.init_inputs = inputs
        self.init_markers = markers
        X_dev, X_val = train_test_split(self.inputs, test_size=0.05, random_state=1, shuffle=True)
        X_train, X_test = train_test_split(X_dev, test_size=0.25, random_state=1)

        self.init_X_train = X_train
        self.init_X_test = X_test
        self.init_X_val = X_val

        # Store the normalized data
        self.markers = self.markers
        self.inputs = np.array(inputs)
        self.X_train = normalize(X_train)
        self.X_test = normalize(X_test)
        self.X_val = normalize(X_val)
        self.inputs_dim = self.inputs.shape[1]
