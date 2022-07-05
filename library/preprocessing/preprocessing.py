import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd
from typing import List, Tuple, Dict


class Preprocessing:

    @staticmethod
    def normalize(data: pd.DataFrame, set_negative_to_zero: bool = False, columns: list = None,
                  standard_scaler: StandardScaler = None) -> (pd.DataFrame, StandardScaler):
        """
        Normalizes the data. Mean is close to 0 and std is close to 1
        @param data:
        @param set_negative_to_zero:
        @param standard_scaler
        @return:
        """
        # Input data contains some zeros which results in NaN (or Inf)
        # values when their log10 is computed. NaN (or Inf) are problematic
        # values for downstream analysis. Therefore, zeros are replaced by
        # a small value; see the following thread for related discussion.
        # https://www.researchgate.net/post/Log_transformation_of_values_that_include_0_zero_for_statistical_analyses2

        data[data == 0] = 1e-32

        if set_negative_to_zero:
            data[data < 0] = 1e-32

        data = np.log10(data)

        if standard_scaler is None:
            standard_scaler = StandardScaler()
            standard_scaler.fit(data)

        data = standard_scaler.transform(data)

        return data, standard_scaler if columns is None else pd.DataFrame(columns=columns, data=data), standard_scaler

    @staticmethod
    def standardize_feature_engineered_data(data: pd.DataFrame, scaler: StandardScaler = None):
        """
        Standardize the data. Mean is close to 0 and std is close to 1
        @param data:
        @param scaler:
        @return:
             """
        # Input data contains some zeros which results in NaN (or Inf)
        # values when their log10 is computed. NaN (or Inf) are problematic
        # values for downstream analysis. Therefore, zeros are replaced by
        # a small value; see the following thread for related discussion.
        # https://www.researchgate.net/post/Log_transformation_of_values_that_include_0_zero_for_statistical_analyses2

        initial_columns: List = list(data.columns)
        data.reset_index(drop=True, inplace=True)
        temp_df = data.copy()

        data = data._get_numeric_data()

        # ordinal_data: List = ['# of Immune Cells', '# of Neoplastic Epithelial Cells', '# of Stroma Cells']

        one_hot_encoded_phenotype: List = data.filter(regex="Phenotype")
        one_hot_encoded_cell_neighborhood: List = data.filter(regex="Cell Neighborhood")

        # for feature_to_remove in ordinal_data:
        #   if feature_to_remove in data:
        #       data.drop(columns=[feature_to_remove], inplace=True)

        for column in one_hot_encoded_phenotype:
            data.drop(columns=[column], inplace=True)

        for column in one_hot_encoded_cell_neighborhood:
            data.drop(columns=[column], inplace=True)

        numeric_columns: List = list(data.columns)

        mean_columns = data.filter(regex="Mean")

        zero_indexes: Dict = {}
        for column in mean_columns:
            indexes = mean_columns.index[mean_columns[column] == 0].tolist()

            zero_indexes[column] = indexes

        data[data == 0] = 1e-32

        # data = np.log10(data)

        if scaler is None:
            scaler = StandardScaler()
            scaler.fit(data)

        data = scaler.transform(data)
        # data = data.clip(min=-5, max=5)

        remaining_columns = list(set(initial_columns) - set(numeric_columns))

        normalized_data = pd.DataFrame(columns=numeric_columns, data=data).copy().sort_index()

        # for column in normalized_data.columns:
        #    if column in zero_indexes.keys():
        #        for index in zero_indexes[column]:
        #            normalized_data.at[index, column] = 0

        remaining_data = temp_df[remaining_columns].copy().sort_index()
        return pd.concat([normalized_data, remaining_data], axis=1), scaler

    @staticmethod
    def normalize_feature_engineered_data(data: pd.DataFrame, scaler: MinMaxScaler = None):
        """
        Normalizes the data. Min is 0, max is 1
        @param data:
        @param min_max_scaler:
        @return:
        """
        # Input data contains some zeros which results in NaN (or Inf)
        # values when their log10 is computed. NaN (or Inf) are problematic
        # values for downstream analysis. Therefore, zeros are replaced by
        # a small value; see the following thread for related discussion.
        # https://www.researchgate.net/post/Log_transformation_of_values_that_include_0_zero_for_statistical_analyses2

        initial_columns: List = list(data.columns)
        data.reset_index(drop=True, inplace=True)
        temp_df = data.copy()

        data = data._get_numeric_data()

        # ordinal_data: List = ['# of Immune Cells', '# of Neoplastic Epithelial Cells', '# of Stroma Cells']
        one_hot_encoded_phenotype: List = data.filter(regex="Phenotype")
        one_hot_encoded_cell_neighborhood: List = data.filter(regex="Cell Neighborhood")

        # for feature_to_remove in ordinal_data:
        #   if feature_to_remove in data:
        #       data.drop(columns=[feature_to_remove], inplace=True)

        for column in one_hot_encoded_phenotype:
            data.drop(columns=[column], inplace=True)

        for column in one_hot_encoded_cell_neighborhood:
            data.drop(columns=[column], inplace=True)

        numeric_columns: List = list(data.columns)

        mean_columns = data.filter(regex="Mean")

        zero_indexes: Dict = {}
        for column in mean_columns:
            indexes = mean_columns.index[mean_columns[column] == 0].tolist()

            zero_indexes[column] = indexes

        data[data == 0] = 1e-32

        # data = np.log10(data)

        if scaler is None:
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaler.fit(data)

        data = scaler.transform(data)
        # data = data.clip(min=-5, max=5)

        remaining_columns = list(set(initial_columns) - set(numeric_columns))

        normalized_data = pd.DataFrame(columns=numeric_columns, data=data).copy().sort_index()

        # for column in normalized_data.columns:
        #    if column in zero_indexes.keys():
        #        for index in zero_indexes[column]:
        #            normalized_data.at[index, column] = 0

        remaining_data = temp_df[remaining_columns].copy().sort_index()
        return pd.concat([normalized_data, remaining_data], axis=1), scaler
