import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np


class Data:
    init_cells: np.array
    init_markers: pd.DataFrame()
    init_X_train = pd.DataFrame()
    init_X_test = pd.DataFrame()
    init_X_val = pd.DataFrame()

    cells: np.array
    markers = pd.DataFrame()
    X_train_noise = pd.DataFrame()
    X_train: pd.DataFrame()
    X_test: pd.DataFrame()
    X_val: pd.DataFrame()

    def __init__(self, cells: np.array, markers: list, normalize):
        """
        Initializes a new object of data
        @param normalize:
        @param inputs:
        @param markers:
        """
        self.init_cells = cells
        self.init_markers = markers
        X_dev, X_val = train_test_split(self.init_cells, test_size=0.05, random_state=1, shuffle=True)
        X_train, X_test = train_test_split(X_dev, test_size=0.25, random_state=1, shuffle=True)

        self.init_X_train = X_train
        self.init_X_test = X_test
        self.init_X_val = X_val

        # Store the normalized data
        self.markers = self.init_markers
        self.cells = normalize(np.array(self.init_cells))
        self.X_train = normalize(X_train)
        self.X_test = normalize(X_test)
        self.X_val = normalize(X_val)
        self.inputs_dim = self.cells.shape[1]
