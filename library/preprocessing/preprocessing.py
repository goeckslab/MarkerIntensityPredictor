import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd


class Preprocessing:
    @staticmethod
    def normalize(data: pd.DataFrame, set_negative_to_zero: bool = False, columns: list = None):
        """
        Normalizes the data. Mean is close to 0 and std is close to 1
        @param data:
        @param set_negative_to_zero:
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

        standard_scaler = StandardScaler()
        data = standard_scaler.fit_transform(data)
        data = data.clip(min=-5, max=5)

        # min_max_scaler = MinMaxScaler(feature_range=(0, 1))
        #  data = min_max_scaler.fit_transform(data)

        return data if columns is None else pd.DataFrame(columns=columns, data=data)
