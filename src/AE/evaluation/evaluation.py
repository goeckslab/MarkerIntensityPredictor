from Shared.data import Data
import pandas as pd
from sklearn.metrics import r2_score
from pathlib import Path
import mlflow


class Evaluation:
    # R2 scores
    r2_scores = pd.DataFrame(columns=["Marker", "Score"])

    __base_result_path: Path
    __base_sub_folder: str
    # The auto encoder model
    __model = None
    # The data the models and everything worked with
    __data: Data

    def __init__(self, base_result_path: Path, model, data: Data):
        self.__base_result_path = base_result_path
        self.__base_sub_folder = "Evaluation"
        self.__model = model
        self.__data = data

    def calculate_r2_score(self):
        recon_test = self.__model.predict(self.__data.X_test)
        recon_test = pd.DataFrame(data=recon_test, columns=self.__data.markers)
        input_data = pd.DataFrame(data=self.__data.X_test, columns=self.__data.markers)

        for marker in self.__data.markers:
            input_marker = input_data[f"{marker}"]
            var_marker = recon_test[f"{marker}"]

            score = r2_score(input_marker, var_marker)
            self.r2_scores = self.r2_scores.append(
                {
                    "Marker": marker,
                    "Score": score
                }, ignore_index=True
            )

        save_path = Path(self.__base_result_path, "r2_scores.csv")
        self.r2_scores.to_csv(save_path, index=False)
        mlflow.log_artifact(str(save_path), str(self.__base_sub_folder))