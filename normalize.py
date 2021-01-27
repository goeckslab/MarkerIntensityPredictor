from sklearn.preprocessing import StandardScaler
import numpy as np


def normalize(data):
    """
    Normalize Data
    :param data:
    :return:
    """
    # Input data contains some zeros which results in NaN (or Inf)
    # values when their log10 is computed. NaN (or Inf) are problematic
    # values for downstream analysis. Therefore, zeros are replaced by
    # a small value; see the following thread for related discussion.
    # https://www.researchgate.net/post/Log_transformation_of_values_that_include_0_zero_for_statistical_analyses2
    data = data + 1e-32
    data = np.log10(data)

    scaler = StandardScaler()
    data = scaler.fit_transform(data)

    # Remove outlier using interquartile range
    q1 = np.quantile(data, 0.25)
    q3 = np.quantile(data, 0.75)
    iqr = q3 - q1
    data = data[~((data < (q1 - 1.5 * iqr)) |(data > (q3 + 1.5 * iqr))).any(axis=1)]

    # A simple max normalization, may worth trying
    # alternative normalization methods
    # (e.g., 0-1 normalization)
    # data = data / data.max(axis=0)
    return data
