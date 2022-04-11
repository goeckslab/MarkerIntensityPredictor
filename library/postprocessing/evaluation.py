import pandas as pd
from sklearn.metrics import r2_score


class PerformanceEvaluator:
    @staticmethod
    def calculate_r2_scores(features: list, ground_truth_data: pd.DataFrame,
                            compare_data: pd.DataFrame) -> pd.DataFrame:
        """
        Creates a dataframe containing the r2 scores for all features provided by the feature list
        @param features:
        @param ground_truth_data:
        @param compare_data:
        @return:
        """
        r2_scores: pd.DataFrame = pd.DataFrame()

        for feature in features:
            ground_truth_feature = ground_truth_data[f"{feature}"]
            compare_feature = compare_data[f"{feature}"]

            score = r2_score(ground_truth_feature, compare_feature)
            r2_scores = r2_scores.append(
                {
                    "Marker": feature,
                    "Score": score
                }, ignore_index=True
            )

        return r2_scores
