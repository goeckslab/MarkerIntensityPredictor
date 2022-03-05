import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import mlflow
from folder_management.folder_management import FolderManagement


class Evaluation:
    # The base path used for the application
    __base_path: Path
    # The results path where all evaluation things are temporarily stored
    __results_path: Path

    def __init__(self, base_path: Path):
        print("Started evaluation...")
        self.__base_path = base_path
        self.__results_path = FolderManagement.create_directory(Path(self.__base_path, "evaluation"))

    def r2_scores_mean_values(self, ae_scores: pd.DataFrame, vae_scores: pd.DataFrame):
        fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(19, 20), dpi=300, sharex=False)
        sns.barplot(x='Marker', y='Score', data=ae_scores, ax=ax1)
        sns.barplot(x='Marker', y='Score', data=vae_scores, ax=ax2)

        # Create difference dataframe
        differences = pd.DataFrame(columns=["Marker", "Difference"], data={"Marker": ae_scores["Marker"].values,
                                                                           "Difference": vae_scores["Score"] -
                                                                                         ae_scores[
                                                                                             "Score"]})
        sns.barplot(x="Marker", y="Difference", data=differences, ax=ax3)

        ax1.set_title("AE R2 Scores")
        ax2.set_title("VAE R2 Scores")
        ax3.set_title("Difference VAE to AE")
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=90)
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=90)
        ax3.set_xticklabels(ax3.get_xticklabels(), rotation=90)

        fig.tight_layout()
        save_path = Path(self.__results_path, "mean_r2_comparison.png")
        plt.savefig(save_path)
        mlflow.log_artifact(str(save_path))
        plt.close()
