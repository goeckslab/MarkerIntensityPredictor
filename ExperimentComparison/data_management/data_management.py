import pandas as pd
from io import BytesIO
from folder_management.folder_management import FolderManagement
from pathlib import Path
import mlflow


class DataManagement:
    client = mlflow.tracking.MlflowClient()
    __base_path: Path

    def __init__(self, root_path: Path):
        self.__base_path = Path(root_path)

    def download_artifacts(self, runs: []):
        """
        Downloads all artifacts of the found experiments
        @return:
        """
        # Create temp directory

        download_directory = FolderManagement.create_directory(Path(self.__base_path, "runs"))
        ae_directory = FolderManagement.create_directory(Path(download_directory, "ae"))
        vae_directory = FolderManagement.create_directory(Path(download_directory, "vae"))

        for run in runs:

            try:
                parent_id_tag = run.data.tags.get('mlflow.parentRunId')

                if "Model" not in run.data.tags and parent_id_tag is None:
                    continue

                model = run.data.tags.get("Model")

                if model == "AE":
                    run_directory = FolderManagement.create_directory(Path(ae_directory, run.info.run_id))

                elif model == "VAE":
                    run_directory = FolderManagement.create_directory(Path(vae_directory, run.info.run_id))

                else:
                    print(f"Model {model} is not implemented. Skipping... ")
                    continue

                DataManagement.client.download_artifacts(run.info.run_id, "Evaluation",
                                                         str(Path(run_directory)))
            except BaseException as ex:
                print(ex)
                continue

    def load_r2_scores_for_model(self, model: str) -> pd.DataFrame:
        """
        Loads all r2scores for the given model and combines them in a dataset
        @param model:
        @return:
        """
        combined_r2_scores = pd.DataFrame()
        markers = []

        path = Path(self.__base_path, "runs", model)
        for p in path.rglob("*"):
            if p.name == "r2_scores.csv":
                df = pd.read_csv(p.absolute(), header=0)

                # Get markers
                markers = df["Marker"].to_list()

                # Transponse
                df = df.T
                # Drop markers row
                df.drop(index=df.index[0],
                        axis=0,
                        inplace=True)

                combined_r2_scores = combined_r2_scores.append(df, ignore_index=True)

        combined_r2_scores.columns = markers

        save_path = Path(self.__base_path, "runs", f"combined_scores_{model}.csv")
        # Save combined data
        combined_r2_scores.to_csv(save_path)
        mlflow.log_artifact(str(save_path))

        # Save mean values
        save_path = Path(self.__base_path, "runs", f"mean_scores_{model}.csv")
        mean_scores = pd.DataFrame(columns=["Marker", "Score"],
                                   data={"Marker": combined_r2_scores.columns,
                                         "Score": combined_r2_scores.mean().values})
        mean_scores.to_csv(save_path, index=False)
        mlflow.log_artifact(str(save_path))

        # return mean scores
        return mean_scores
