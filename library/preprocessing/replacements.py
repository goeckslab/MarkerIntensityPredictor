import pandas as pd
import numpy as np
from typing import Tuple


class Replacer:

    @staticmethod
    def replace_values(data: pd.DataFrame, feature_to_replace: str, percentage: float,
                       value_to_replace_with: any) -> Tuple[pd.DataFrame, list]:
        """
        Replaces a specific
        @param data:
        @param feature_to_replace:
        @param percentage:
        @param value_to_replace_with:
        @return:
        """
        # Replace % of the data provided by the args
        data[feature_to_replace] = data[feature_to_replace].sample(frac=1 - percentage, replace=False).copy()

        indexes = data[data[feature_to_replace].isna()].index

        if value_to_replace_with == 0:
            values = [0] * data[feature_to_replace].isna().sum()
            data[feature_to_replace].fillna(pd.Series(values, index=indexes), inplace=True)
            return data, indexes

        if value_to_replace_with is np.nan:
            return data, indexes
