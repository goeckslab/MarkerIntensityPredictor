import pandas as pd
from typing import Tuple
from sklearn.model_selection import train_test_split


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
