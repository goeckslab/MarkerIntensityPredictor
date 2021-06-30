import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class DataStorage:
    inputs = pd.DataFrame()
    markers: list
    X_dev = pd.DataFrame()
    X_val = pd.DataFrame()
    X_train = pd.DataFrame()
    X_test = pd.DataFrame()

    init_inputs = pd.DataFrame()
    init_X_train = pd.DataFrame()
    init_X_test = pd.DataFrame()
    init_X_val = pd.DataFrame()

    def __init__(self, inputs, markers):
        self.inputs = inputs
        self.markers = markers

        self.inputs = np.array(self.inputs)
        self.X_dev, self.X_val = train_test_split(self.inputs, test_size=0.05, random_state=1, shuffle=True)
        self.X_train, self.X_test = train_test_split(self.X_dev, test_size=0.25, random_state=1)

        self.init_inputs = self.inputs
        self.init_X_train = self.X_train
        self.init_X_test = self.X_test
        self.init_X_val = self.X_val

        self.inputs = pd.DataFrame(columns=self.markers, data=self.normalize(self.inputs))
        self.X_train = pd.DataFrame(columns=self.markers, data=self.normalize(self.X_train))
        self.X_test = pd.DataFrame(columns=self.markers, data=self.normalize(self.X_test))
        self.X_val = pd.DataFrame(columns=self.markers, data=self.normalize(self.X_val))

    @staticmethod
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
        data = data.clip(min=-5, max=5)

        # min_max_scaler = MinMaxScaler()
        # data = min_max_scaler.fit_transform(data)
        return data
