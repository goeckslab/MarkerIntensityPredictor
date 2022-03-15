import pandas as pd
from sklearn.metrics import r2_score
from pathlib import Path
import mlflow


class Evaluation:

    @staticmethod
    def calculate_r2_score(ground_truth: pd.DataFrame, reconstructed_data: pd.DataFrame, markers: list) -> pd.DataFrame:
        """
        Calculates the r2 scores for the given parameters
        @param ground_truth: The input data to evaluate
        @param reconstructed_data: The reconstructed data to evaluate
        @param markers: The markers of the dataset
        @return: Returns a dataframe containing all r2 scores
        """
        r2_scores = pd.DataFrame(columns=["Marker", "Score"])
        recon_test = pd.DataFrame(data=reconstructed_data, columns=markers)
        test_data = pd.DataFrame(data=ground_truth, columns=markers)

        for marker in markers:
            ground_truth_marker = test_data[f"{marker}"]
            reconstructed_marker = recon_test[f"{marker}"]

            score = r2_score(ground_truth_marker, reconstructed_marker)
            r2_scores = r2_scores.append(
                {
                    "Marker": marker,
                    "Score": score
                }, ignore_index=True
            )

        return r2_scores
