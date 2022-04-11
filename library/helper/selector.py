import pandas as pd


class Selector:

    @staticmethod
    def select_rows_by_index_and_feature(data_set: pd.DataFrame, feature: str, indexes: list) -> pd.DataFrame:
        return data_set[feature].iloc[indexes].copy()
