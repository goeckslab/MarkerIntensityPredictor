import pandas as pd
from sklearn.metrics import r2_score
from pathlib import Path
import mlflow


class Evaluation:
    # R2 scores

    @staticmethod
    def calculate_r2_score(test_data: pd.DataFrame, reconstructed_data: pd.DataFrame, markers: list) -> pd.DataFrame:
        """
        Calculates the r2 scores for the given parameters
        @param test_data: The input data to evaluate
        @param reconstructed_data: The reconstructed data to evaluate
        @param markers: The markers of the dataset
        @param save_path: Where to store the file?
        @param mlflow_folder: If given, this is the subfolder in the mlflow directory
        @return:
        """
        r2_scores = pd.DataFrame(columns=["Marker", "Score"])
        recon_test = pd.DataFrame(data=reconstructed_data, columns=markers)
        test_data = pd.DataFrame(data=test_data, columns=markers)

        for marker in markers:
            input_marker = test_data[f"{marker}"]
            var_marker = recon_test[f"{marker}"]

            score = r2_score(input_marker, var_marker)
            r2_scores = r2_scores.append(
                {
                    "Marker": marker,
                    "Score": score
                }, ignore_index=True
            )



        return r2_scores
