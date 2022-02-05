import numpy as np
from sklearn.preprocessing import StandardScaler


def normalize(inputs: np.ndarray):
    # Input data contains some zeros which results in NaN (or Inf)
    # values when their log10 is computed. NaN (or Inf) are problematic
    # values for downstream analysis. Therefore, zeros are replaced by
    # a small value; see the following thread for related discussion.
    # https://www.researchgate.net/post/Log_transformation_of_values_that_include_0_zero_for_statistical_analyses2

    inputs[inputs == 0] = 1e-32
    inputs = np.log10(inputs)

    standard_scaler = StandardScaler()
    inputs = standard_scaler.fit_transform(inputs)
    inputs = inputs.clip(min=-5, max=5)

    # min_max_scaler = MinMaxScaler(feature_range=(-1, 1))
    # inputs = min_max_scaler.fit_transform(inputs)

    return inputs
