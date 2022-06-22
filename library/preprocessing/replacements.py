import pandas as pd
import numpy as np
from typing import Tuple, List, Dict
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
    def select_index_and_features_to_replace(features: List, length_of_data: int, percentage: float) -> Dict:
        """
        Based on the available features and the length of the data, features will be selected for each row
        @param features:
        @param length_of_data:
        @param percentage:
        @return: A dict containing the index and the features which should be replaced
        """
        available_features = copy.deepcopy(features)

        if "Y_centroid" in available_features:
            available_features.remove("Y_centroid")
        if "X_centroid" in available_features:
            available_features.remove("X_centroid")

        replaced_feature_per_index: dict = {}
        amount_to_replace = 1 if int(len(available_features) * percentage) == 0 else int(
            len(available_features) * percentage)

        for index in range(length_of_data):
            features_to_replace: list = random.sample(available_features, amount_to_replace)
            replaced_feature_per_index[index] = features_to_replace

        return replaced_feature_per_index

    @staticmethod
    def replace_values_by_cell(data: pd.DataFrame, index_replacements: Dict, value_to_replace=np.nan) -> pd.DataFrame:
        df = data.copy()

        for key, values in index_replacements.items():
            df.at[key, values] = value_to_replace

        return df

    @staticmethod
    def load_index_replacement_file(file_path: str) -> Dict:
        index_replacements_df: pd.DataFrame = pd.read_csv(file_path)
        values: List = index_replacements_df.values.tolist()
        index_replacements: Dict = {}
        # Convert dataframe back to expected dictionary
        for i, value in enumerate(values):
            index_replacements[i] = value

        return index_replacements
