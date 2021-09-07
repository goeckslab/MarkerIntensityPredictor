import pandas as pd
from sklearn.model_selection import train_test_split


class Data:
    init_markers: pd.DataFrame()
    init_X_train = pd.DataFrame()
    init_X_test = pd.DataFrame()
    init_X_val = pd.DataFrame()

    markers = pd.DataFrame()
    X_train_noise = pd.DataFrame()
    X_train: pd.DataFrame()
    X_test: pd.DataFrame()
    X_val: pd.DataFrame()

    def __init__(self, train, test, markers, normalize):
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
