import pandas as pd
from sklearn.impute import KNNImputer
from typing import List, Dict
from sklearn.metrics import r2_score


class KNNImputation:

    @staticmethod
    def impute(train_data: pd.DataFrame, test_data: pd.DataFrame, missing_values: any = 0,
               n_neighbors: int = 2) -> pd.DataFrame:
        imputer = KNNImputer(n_neighbors=n_neighbors, missing_values=missing_values)
        imputer.fit(train_data)
        imputed_values = imputer.transform(test_data)

        return pd.DataFrame(data=imputed_values, columns=test_data.columns)

    @staticmethod
    def fit_imputer(train_data: pd.DataFrame, missing_values: any = 0, n_neighbors: int = 2) -> KNNImputer:
        imputer = KNNImputer(n_neighbors=n_neighbors, missing_values=missing_values)
        imputer.fit(train_data)
        return imputer

    @staticmethod
    def evaluate_performance(features: List, index_replacements: Dict, test_data: pd.DataFrame,
                             imputed_data: pd.DataFrame):
        """
        Evaluates the performance per feature
        @param features:
        @param index_replacements:
        @param test_data:
        @param imputed_data:
        @return: Returns a dataframe with r2 scores for performance evaluation
        """
        score_data: List = []

        for feature in features:
            if "X_centroid" in feature or "Y_centroid" in feature:
                continue

            # Store all cell indexes, to be able to select the correct cells later for r2 comparison
            cell_indexes_to_compare: list = []
            for key, replaced_features in index_replacements.items():
                if feature in replaced_features:
                    cell_indexes_to_compare.append(key)

            score_data.append({
                "Feature": feature,
                "Score": r2_score(test_data[feature].iloc[cell_indexes_to_compare],
                                  imputed_data[feature].iloc[cell_indexes_to_compare])
            })

        return pd.DataFrame().from_records(score_data)
