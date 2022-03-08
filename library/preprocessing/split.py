import pandas as pd
from typing import Tuple
from sklearn.model_selection import train_test_split, KFold


def create_splits(cells: pd.DataFrame, without_val: bool = False) -> Tuple:
    """
    Creates train val test split of the data provided
    @param cells:
    @param without_val:
    @return: A tuple containing all the test sets.
    """

    # No validation set will be created
    if without_val:
        return train_test_split(cells, test_size=0.2, random_state=1, shuffle=True)

    # Create validation set
    X_dev, X_val = train_test_split(cells, test_size=0.05, random_state=1, shuffle=True)
    X_train, X_test = train_test_split(X_dev, test_size=0.25, random_state=1, shuffle=True)
    return X_train, X_val, X_test


def create_folds(data_to_split: pd.DataFrame, splits: int = 5) -> Tuple:
    # https://datascience.stackexchange.com/a/52643
    kf = KFold(n_splits=splits, random_state=1, shuffle=True)
    kf.get_n_splits(data_to_split)

    for train_index, test_index in kf.split(data_to_split):
        yield data_to_split[train_index], data_to_split[test_index]
