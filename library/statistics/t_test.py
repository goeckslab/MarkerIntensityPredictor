import pandas as pd
from scipy import stats
from typing import List


class TTest:

    @staticmethod
    def calculateTTest(data: pd.DataFrame, data_to_compare: pd.DataFrame, column: str,
                       column_value: str) -> pd.DataFrame:
        """
        Calculates a t-test for the given dataframes
        @param data:
        @param data_to_compare:
        @param column:
        @param column_value:
        @return:
        """
        t_test = stats.ttest_ind(data, data_to_compare)

        t_test_data: List = [{
            column: column_value,
            "T-Value": t_test[0],
            "p-Value": t_test[1]

        }]

        return pd.DataFrame.from_records(t_test_data)
