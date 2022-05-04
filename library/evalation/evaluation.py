import pandas as pd
from sklearn.metrics import r2_score


class Evaluation:

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

    @staticmethod
    def create_relative_score_performance(r2_scores: dict, features: list, reference_model: str) -> pd.DataFrame:
        relative_score_performance: pd.DataFrame = pd.DataFrame()
        reference_r2_scores = r2_scores.get(reference_model)
        if reference_r2_scores is None:
            raise ValueError(f"{reference_model} is not stored in dict")

        for feature in features:

            base_performance = reference_r2_scores.loc[reference_r2_scores['Marker'] == feature]["Score"].tolist()[0]

            for model in r2_scores.keys():
                model_scores = r2_scores.get(model)
                model_performance = model_scores.loc[model_scores['Marker'] == feature]["Score"].tolist()[0]

                # Calculate relative performance
                relative_performance = model_performance / base_performance if base_performance > 0 else 0

                relative_score_performance = relative_score_performance.append({
                    "Feature": feature,
                    "Score": relative_performance,
                    "Model": model,
                }, ignore_index=True)

        return relative_score_performance

    @staticmethod
    def create_absolute_score_performance(r2_scores: dict, features: list) -> pd.DataFrame:
        """
        Converts the data into the format required to use it for further plotting
        @param r2_scores:
        @param features:
        @return:
        """
        absolute_score_performance: pd.DataFrame = pd.DataFrame()
        for feature in features:
            for model in r2_scores.keys():
                model_scores = r2_scores.get(model)

                if feature in model_scores["Marker"].tolist():
                    model_performance = model_scores.loc[model_scores['Marker'] == feature]["Score"].tolist()[0]
                else:
                    model_performance = 0

                absolute_score_performance = absolute_score_performance.append({
                    "Feature": feature,
                    "Score": model_performance,
                    "Model": model,
                }, ignore_index=True)

        return absolute_score_performance
