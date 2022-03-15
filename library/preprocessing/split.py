import numpy as np
import pandas as pd
from typing import Tuple, Optional
from sklearn.model_selection import train_test_split, KFold


def create_splits(cells: pd.DataFrame, create_val: bool = True, seed: int = 1) -> Tuple:
    """
    Creates train val test split of the data provided
    @param cells: The cells to split
    @param create_val: Should a validation set be created?
    @param seed: The seed to use
    @return: A tuple containing all the test sets.
    """

    # No validation set will be created
    if not create_val:
        return train_test_split(cells, test_size=0.2, random_state=seed, shuffle=True)

    # Create validation set
    X_dev, X_val = train_test_split(cells, test_size=0.05, random_state=seed, shuffle=True)
    X_train, X_test = train_test_split(X_dev, test_size=0.25, random_state=seed, shuffle=True)
    return X_train, X_val, X_test


def create_folds(data_to_split: pd.DataFrame, splits: int = 5, seed: int = 1) -> Tuple:
    # https://datascience.stackexchange.com/a/52643
    kf = KFold(n_splits=splits, random_state=seed, shuffle=True)
    kf.get_n_splits(data_to_split)

    for train_index, test_index in kf.split(data_to_split):
        yield data_to_split[train_index], data_to_split[test_index]
