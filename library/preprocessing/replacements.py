import pandas as pd
import numpy as np
from typing import Tuple
import random
import copy


class Replacer:

    @staticmethod
    def replace_values_by_feature(data: pd.DataFrame, feature_to_replace: str, percentage: float,
                                  value_to_replace_with: any = 0) -> Tuple[pd.DataFrame, list]:
        """
        Replaces a specific
        @param data:
        @param feature_to_replace:
        @param percentage:
        @param value_to_replace_with:
        @return: Returns the complete data with replacements and the indexes of the replacements
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

    @staticmethod
    def replace_values_by_cell(data: pd.DataFrame, features: list, percentage: float) -> (pd.DataFrame, dict):
        df = data.copy()
        replaced_feature_per_index: dict = {}

        available_features = copy.deepcopy(features)
        # Remove x and y from features

        if "Y_centroid" in available_features:
            available_features.remove("Y_centroid")
        if "X_centroid" in available_features:
            available_features.remove("X_centroid")

        for index, row in df.iterrows():
            amount_to_replace = 1 if int(len(available_features) * percentage) == 0 else int(
                len(available_features) * percentage)

            features_to_replace: list = random.sample(available_features, amount_to_replace)
            replaced_feature_per_index[index] = features_to_replace
            for feature_to_replace in features_to_replace:
                df.at[index, feature_to_replace] = 0

        return df, replaced_feature_per_index
