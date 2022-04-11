import pandas as pd
from sklearn.impute import KNNImputer


class KNNImputation:

    @staticmethod
    def impute(data_set: pd.DataFrame, missing_values: any = 0, n_neighbors: int = 2) -> pd.DataFrame:
        imputer = KNNImputer(n_neighbors=n_neighbors, missing_values=missing_values)
        imputed_values = imputer.fit_transform(data_set)

        return pd.DataFrame(data=imputed_values, columns=data_set.columns)
