import pandas as pd
from services.data_loader import DataLoader
import numpy as np
from sklearn.model_selection import train_test_split


class MarkerIntensity:
    input_file: str
    inputs = pd.DataFrame()
    markers: list

    def __init__(self, input_file):
        self.input_file = input_file

    def load(self):
        inputs, markers = DataLoader.get_data(self.input_file)
        self.markers = markers
        self.inputs = np.array(inputs)
        X_dev, X_val = train_test_split(inputs, test_size=0.05, random_state=1, shuffle=True)
        X_train, X_test = train_test_split(X_dev, test_size=0.25, random_state=1)

    def get_input_stats(self):
        if self.input_file is None:
            return

    def train(self):
        pass

    def prediction(self):
        pass

    def get_accuracy(self):
        pass
