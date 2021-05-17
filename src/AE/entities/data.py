import pandas as pd
import numpy as np


class Data:
    inputs: np.array
    markers: pd.DataFrame()
    X_train: pd.DataFrame()
    X_test: pd.DataFrame()
    X_val: pd.DataFrame()
