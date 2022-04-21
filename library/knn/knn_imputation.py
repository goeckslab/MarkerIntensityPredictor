import pandas as pd
from sklearn.impute import KNNImputer


class KNNImputation:

    @staticmethod
    def impute(train_data: pd.DataFrame, test_data: pd.DataFrame, missing_values: any = 0,
               n_neighbors: int = 2) -> pd.DataFrame:
        imputer = KNNImputer(n_neighbors=n_neighbors, missing_values=missing_values)
        imputer.fit(train_data)
        imputed_values = imputer.transform(test_data)

        return pd.DataFrame(data=imputed_values, columns=test_data.columns)
