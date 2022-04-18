import logging
import matplotlib.pyplot as plt
from pathlib import Path
import mlflow
import pandas as pd
import seaborn as sns
from itertools import combinations
from typing import Tuple
from tensorflow.keras.utils import plot_model
import numpy as np

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


# Figsize (7,9) First is width, second is height

class Plotting:
    # The base path such as AE or VAE. This is the path where the files will be stored
    __base_path: Path

    def __init__(self, base_path: Path, args):
        self.__base_path = base_path
        if args.tracking_url is not None:
            mlflow.set_tracking_uri = args.tracking_url

    def plot_model_performance(self, history, file_name: str, mlflow_directory: str = None):
        logger.info("Plotting model performance")
        plt.figure(num=None, figsize=(6, 4), dpi=90)
        for key in history.history:
            plt.plot(history.history[key], label=key)
        plt.xlabel("Epoch")
        plt.ylabel("Value")
        plt.legend()
        plt.tight_layout()

        save_path = Path(self.__base_path, f"{file_name}.png")
        plt.savefig(save_path)
        if mlflow_directory is not None:
            mlflow.log_artifact(str(save_path), mlflow_directory)
        else:
            mlflow.log_artifact(str(save_path))
        plt.close()

    def plot_reconstructed_markers(self, test_data, reconstructed_data, markers, mlflow_directory: str,
                                   file_name: str):
        logging.info("Plotting reconstructed intensities")
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 10), dpi=300, sharex=True)
        sns.heatmap(test_data, ax=ax1, xticklabels=markers)
        sns.heatmap(reconstructed_data, ax=ax2, xticklabels=markers)

        ax1.set_title("X Test")
        ax2.set_title("Reconstructed X Test")
        fig.tight_layout()

        save_path = Path(self.__base_path, f"{file_name}.png")
        plt.savefig(save_path)
        mlflow.log_artifact(str(save_path), mlflow_directory)
        plt.close()

    def plot_feature_intensities(self, train_data: pd.DataFrame, test_data: pd.DataFrame, features: list,
                                 mlflow_directory: str, file_name: str, val_data=None):
        if val_data is not None:
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(9, 10), dpi=300, sharex=True)
        else:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 10), dpi=300, sharex=True)

        sns.heatmap(train_data, ax=ax1, xticklabels=features)
        sns.heatmap(test_data, ax=ax2, xticklabels=features)

        if val_data is not None:
            sns.heatmap(val_data, ax=ax3, xticklabels=features)

        ax1.set_title("X Train")
        ax2.set_title("X Test")

        if val_data is not None:
            ax3.set_title("X Validation")

        fig.tight_layout()

        save_path = Path(self.__base_path, f"{file_name}.png")
        plt.savefig(save_path)
        mlflow.log_artifact(str(save_path), mlflow_directory)
        plt.close()

    def plot_weights(self, weights, features: list, file_name: str, mlflow_directory: str = None):
        df = pd.DataFrame(weights, columns=features)
        ax = sns.heatmap(df)

        # Markers count is 26. As all are present we can change the labels
        if len(df) == 25:
            ax.set_yticks(range(len(df)))
            ax.set_yticklabels(features)
            ax.set_xticks(range(len(df)))
            ax.set_xticklabels(features)

        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

        fig = ax.get_figure()
        fig.tight_layout()
        save_path = Path(self.__base_path, f"{file_name}.png")
        plt.savefig(save_path)

        if mlflow_directory is not None:
            mlflow.log_artifact(str(save_path), mlflow_directory)
        else:
            mlflow.log_artifact(str(save_path))
        plt.close()

    def r2_scores_relative_performance(self, relative_score_performance: pd.DataFrame, features: list,
                                       file_name: str,
                                       mlflow_directory: str = None):
        """
        Plots the relative performance of the dataset, using EN as baseline
        @param relative_score_performance:
        @param features:
        @param file_name:
        @param mlflow_directory:
        @return:
        """

        fig, ax = plt.subplots(ncols=1, figsize=(7, 5), dpi=300)
        # Draw a nested barplot by species and sex
        ax = sns.catplot(
            data=relative_score_performance, kind="bar",
            x="Feature", y="Score", hue="Model",
            ci="sd", palette="dark", alpha=.6, height=6
        )
        ax.despine(left=True)
        ax.set_axis_labels("Features", "Relative Difference R2 Score")
        # Rotate labels
        for axes in ax.axes.flat:
            _ = axes.set_xticklabels(axes.get_xticklabels(), rotation=90)

        # Put a legend below current axis
        fig.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
                   fancybox=True, shadow=True, ncol=len(relative_score_performance["Model"].unique()))
        plt.tight_layout()

        save_path = Path(self.__base_path, f"{file_name}.png")
        plt.savefig(save_path)

        if mlflow_directory is not None:
            mlflow.log_artifact(str(save_path), mlflow_directory)
        else:
            mlflow.log_artifact(str(save_path))
        plt.close()

    def r2_scores_absolute_performance(self, absolute_score_performance: pd.DataFrame, file_name: str,
                                       mlflow_directory: str = None):
        """
        Plots the absolute performance difference for the given dataset
        @return:
        """

        fig, ax = plt.subplots(ncols=1, figsize=(7, 5), dpi=300)
        # Draw a nested barplot by species and sex
        ax = sns.catplot(
            data=absolute_score_performance, kind="bar",
            x="Feature", y="Score", hue="Model",
            ci="sd", palette="dark", alpha=.6, height=6
        )
        ax.despine(left=True)
        ax.set_axis_labels("Features", "Relative Difference R2 Score")
        # Rotate labels
        for axes in ax.axes.flat:
            _ = axes.set_xticklabels(axes.get_xticklabels(), rotation=90)

        # Put a legend below current axis
        fig.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
                   fancybox=True, shadow=True, ncol=len(absolute_score_performance["Model"].unique()))
        plt.tight_layout()

        save_path = Path(self.__base_path, f"{file_name}.png")
        plt.savefig(save_path)

        if mlflow_directory is not None:
            mlflow.log_artifact(str(save_path), mlflow_directory)
        else:
            mlflow.log_artifact(str(save_path))
        plt.close()

    def r2_score_differences(self, r2_scores: dict, file_name: str, mlflow_directory: str = None):
        """
        Plots the difference between the scores provided in the dict
        @param r2_scores:
        @param file_name:
        @param mlflow_directory:
        @return:
        """
        if len(r2_scores.items()) < 2:
            return

        possible_combinations: list = list(combinations(r2_scores.keys(), 2))

        # Determine number of rows
        num_rows = 1
        if len(possible_combinations) > 3:
            num_rows = float(len(r2_scores.items()) / 3)
            if not num_rows.is_integer():
                num_rows += 1

            num_rows = int(num_rows)

        n_cols = 3

        # Adjust columns based on items
        if num_rows == 1:
            fig, axs = plt.subplots(ncols=len(r2_scores.keys()), nrows=num_rows, figsize=(12, 7), dpi=300, sharex=False)
        elif num_rows == 2:
            fig, axs = plt.subplots(ncols=3, nrows=num_rows, figsize=(12, 9), dpi=300, sharex=False)
        elif num_rows == 3:
            fig, axs = plt.subplots(ncols=3, nrows=num_rows, figsize=(12, 11), dpi=300, sharex=False)
        else:
            fig, axs = plt.subplots(ncols=3, nrows=num_rows, figsize=(12, 13), dpi=300, sharex=False)

        col: int = 0
        row: int = 0

        combination: Tuple
        for combination in possible_combinations:
            experiment_name: str = combination[0]
            compare_experiment_name: str = combination[1]

            r2_score = r2_scores[combination[0]]
            compare_score = r2_scores[combination[1]]

            # Create difference dataframe
            differences = pd.DataFrame(columns=["Marker", "Score"],
                                       data={"Marker": r2_score["Marker"].values,
                                             "Score": r2_score["Score"] - compare_score["Score"]})

            if num_rows == 1:
                if len(possible_combinations) == 1:
                    sns.barplot(x='Markers', y='Score', data=differences, ax=axs)
                    axs.set_title(f"Difference {experiment_name} vs. {compare_experiment_name}")
                    axs.set_xticklabels(axs.get_xticklabels(), rotation=90)
                    axs.set_ylim(0, 1)
                    col += 1
                else:
                    sns.barplot(x='Marker', y='Score', data=differences, ax=axs[col])
                    axs[col].set_title(f"Difference {experiment_name} vs. {compare_experiment_name}")
                    axs[col].set_xticklabels(axs[col].get_xticklabels(), rotation=90)
                    axs[col].set_ylim(0, 1)
                    col += 1


            else:
                sns.barplot(x='Marker', y='Score', data=differences, ax=axs[row, col])
                axs[row, col].set_title(f"Difference {experiment_name} vs. {compare_experiment_name}")
                axs[row, col].set_xticklabels(axs[row, col].get_xticklabels(), rotation=90)
                axs[row, col].set_ylim(0, 1)
                col += 1

                if col == n_cols:
                    row += 1
                    col = 0

        fig.tight_layout()
        save_path = Path(self.__base_path, f"{file_name}.png")
        plt.savefig(save_path)

        if mlflow_directory is not None:
            mlflow.log_artifact(str(save_path), mlflow_directory)
        else:
            mlflow.log_artifact(str(save_path))
        plt.close()

    def plot_scores(self, scores: dict, file_name: str, mlflow_directory: str = None):
        """
        Creates a bar plot for all provided score values
        @param scores: A dict which contains all r2 scores. Keys are used as sub plot titles. Values are scores
        @param file_name: The file name to use for storing the file
        @param mlflow_directory: The mlflow directory to save the file
        @param prefix: An optional prefix for the file
        @return:
        """
        num_rows = 1
        if len(scores.items()) > 3:
            num_rows = float(len(scores.items()) / 3)
            if not num_rows.is_integer():
                num_rows += 1

            num_rows = int(num_rows)

        n_cols = 3

        # Adjust columns based on items
        if num_rows == 1:
            fig, axs = plt.subplots(ncols=len(scores.keys()), nrows=num_rows, figsize=(12, 7), dpi=300, sharex=False)
        elif num_rows == 2:
            fig, axs = plt.subplots(ncols=3, nrows=num_rows, figsize=(12, 9), dpi=300, sharex=False)
        elif num_rows == 3:
            fig, axs = plt.subplots(ncols=3, nrows=num_rows, figsize=(12, 11), dpi=300, sharex=False)
        else:
            fig, axs = plt.subplots(ncols=3, nrows=num_rows, figsize=(12, 13), dpi=300, sharex=False)

        col: int = 0
        row: int = 0

        if num_rows == 1:
            for experiment_name, score in scores.items():
                if len(scores.items()) == 1:
                    sns.barplot(x='Marker', y='Score', data=score, ax=axs)
                    axs.set_title(experiment_name)
                    axs.set_xticklabels(axs.get_xticklabels(), rotation=90)
                    axs.set_ylim(0, 1)
                else:
                    sns.barplot(x='Marker', y='Score', data=score, ax=axs[col])
                    axs[col].set_title(experiment_name)
                    axs[col].set_xticklabels(axs[col].get_xticklabels(), rotation=90)
                    axs[col].set_ylim(0, 1)
                    col += 1

        else:
            for experiment_name, score in scores.items():
                sns.barplot(x='Marker', y='Score', data=score, ax=axs[row, col])
                axs[row, col].set_title(experiment_name)
                axs[row, col].set_xticklabels(axs[row, col].get_xticklabels(), rotation=90)
                axs[row, col].set_ylim(0, 1)
                col += 1

                if col == n_cols:
                    row += 1
                    col = 0

        fig.tight_layout()

        save_path = Path(self.__base_path, f"{file_name}.png")

        plt.savefig(save_path)

        if mlflow_directory is not None:
            mlflow.log_artifact(str(save_path), mlflow_directory)
        else:
            mlflow.log_artifact(str(save_path))
        plt.close()

    def r2_score_model_distribution(self, vae_r2_scores: dict, ae_r2_scores: dict, en_r2_scores: dict,
                                    me_vae_r2_scores: dict, file_name: str):
        """
        Plots a graph which aligns all models next to each other for each experiment
        @param vae_r2_scores: The vae scores
        @param ae_r2_scores: The ae scores
        @param en_r2_scores: The en scores
        @param me_vae_r2_scores: The en scores
        @param file_name: The file name which should be used for the file
        @return:
        """

        if len(vae_r2_scores.values()) < 1 or len(ae_r2_scores.values()) < 1 or len(en_r2_scores.values()) < 1 or len(
                me_vae_r2_scores.values()) < 1:
            print("Dictionaries do not contain enough values. Please make sure all keys are included for all models.")
            return

        if len(vae_r2_scores.keys()) != len(ae_r2_scores.keys()) != len(en_r2_scores.keys()) != len(
                me_vae_r2_scores.keys()):
            print("Dictionaries do not contain the same amount of experiments.")
            return

        num_rows = int(len(vae_r2_scores.keys()))

        if num_rows == 1:
            fig, axs = plt.subplots(ncols=4, nrows=num_rows, figsize=(12, 7), dpi=300, sharex=False)
        elif num_rows == 2:
            fig, axs = plt.subplots(ncols=4, nrows=num_rows, figsize=(12, 9), dpi=300, sharex=False)
        elif num_rows == 3:
            fig, axs = plt.subplots(ncols=4, nrows=num_rows, figsize=(12, 11), dpi=300, sharex=False)
        else:
            fig, axs = plt.subplots(ncols=4, nrows=num_rows, figsize=(12, 13), dpi=300, sharex=False)

        col: int = 0
        row: int = 0

        experiment_names: [] = [experiment_name for experiment_name in vae_r2_scores.keys()]

        for experiment_name in experiment_names:
            vae_r2_score = vae_r2_scores[experiment_name]
            ae_r2_score = ae_r2_scores[experiment_name]
            en_r2_score = en_r2_scores[experiment_name]
            me_vae_r2_score = me_vae_r2_scores[experiment_name]

            sns.boxplot(data=vae_r2_score, ax=axs[row, col])
            axs[row, col].set_title(f"{experiment_name} EN")
            axs[row, col].set_xticklabels(axs[row, col].get_xticklabels(), rotation=90)
            axs[row, col].set_ylim(0, 1)
            col += 1

            sns.boxplot(data=ae_r2_score, ax=axs[row, col])
            axs[row, col].set_title(f"{experiment_name} AE")
            axs[row, col].set_xticklabels(axs[row, col].get_xticklabels(), rotation=90)
            axs[row, col].set_ylim(0, 1)
            col += 1

            sns.boxplot(data=en_r2_score, ax=axs[row, col])
            axs[row, col].set_title(f"{experiment_name} VAE")
            axs[row, col].set_xticklabels(axs[row, col].get_xticklabels(), rotation=90)
            axs[row, col].set_ylim(0, 1)
            col += 1

            sns.boxplot(data=me_vae_r2_score, ax=axs[row, col])
            axs[row, col].set_title(f"{experiment_name} ME VAE")
            axs[row, col].set_xticklabels(axs[row, col].get_xticklabels(), rotation=90)
            axs[row, col].set_ylim(0, 1)
            col += 1

            row += 1
            col = 0

        fig.tight_layout()
        save_path = Path(self.__base_path, f"{file_name}.png")
        plt.savefig(save_path)
        mlflow.log_artifact(str(save_path))
        plt.close()

    def r2_score_model_distribution_comparison(self, r2_scores: dict, compare_r2_scores: dict, model: str,
                                               model_to_compare: str, file_name: str):
        """
        Plots a graph which aligns all models next to each other for each experiment
        @param r2_scores: The vae scores
        @param compare_r2_scores: The en scores
        @param file_name: The file name which should be used for the file
        @return:
        """

        if len(r2_scores.values()) < 1 or len(compare_r2_scores.values()) < 1:
            print("Dictionaries do not contain enough values. Please make sure all keys are included for all models.")
            return

        if len(r2_scores.keys()) != len(compare_r2_scores.keys()):
            print("Dictionaries do not contain the same amount of experiments.")
            return

        num_rows = int(len(compare_r2_scores.keys()))

        if num_rows == 1:
            fig, axs = plt.subplots(ncols=2, nrows=num_rows, figsize=(12, 7), dpi=300, sharex=False)
        elif num_rows == 2:
            fig, axs = plt.subplots(ncols=2, nrows=num_rows, figsize=(12, 9), dpi=300, sharex=False)
        elif num_rows == 3:
            fig, axs = plt.subplots(ncols=2, nrows=num_rows, figsize=(12, 11), dpi=300, sharex=False)
        else:
            fig, axs = plt.subplots(ncols=2, nrows=num_rows, figsize=(12, 13), dpi=300, sharex=False)

        col: int = 0
        row: int = 0

        experiment_names: [] = [experiment_name for experiment_name in r2_scores.keys()]

        for experiment_name in experiment_names:
            r2_score = r2_scores[experiment_name]
            compare_r2_score = compare_r2_scores[experiment_name]

            sns.boxplot(data=r2_score, ax=axs[row, col])
            axs[row, col].set_title(f"{experiment_name} {model}")
            axs[row, col].set_xticklabels(axs[row, col].get_xticklabels(), rotation=90)
            axs[row, col].set_ylim(0, 1)
            col += 1

            sns.boxplot(data=compare_r2_score, ax=axs[row, col])
            axs[row, col].set_title(f"{experiment_name} {model_to_compare}")
            axs[row, col].set_xticklabels(axs[row, col].get_xticklabels(), rotation=90)
            axs[row, col].set_ylim(0, 1)
            col += 1

            row += 1
            col = 0

        fig.tight_layout()
        save_path = Path(self.__base_path, f"{file_name}.png")
        plt.savefig(save_path)
        mlflow.log_artifact(str(save_path))
        plt.close()

    def r2_score_model_mean(self, vae_r2_scores: dict, ae_r2_scores: dict, en_r2_scores: dict, me_vae_r2_scores: dict,
                            file_name: str):
        """
        Plots a graph which aligns all models next to each other for each experiment
        @param vae_r2_scores: The vae scores
        @param ae_r2_scores: The ae scores
        @param en_r2_scores: The en scores
        @param file_name: The file name which should be used for the file
        @return:
        """

        if len(vae_r2_scores.values()) < 1 or len(ae_r2_scores.values()) < 1 or len(en_r2_scores.values()) < 1:
            print("Dictionaries do not contain enough values. Please make sure all keys are included for all models.")
            return

        if len(vae_r2_scores.keys()) != len(ae_r2_scores.keys()) != len(en_r2_scores.keys()):
            print("Dictionaries do not contain the same amount of experiments.")
            return

        num_rows = int(len(vae_r2_scores.keys()))

        # Adjust figure size
        if num_rows == 1:
            fig, axs = plt.subplots(ncols=4, nrows=num_rows, figsize=(12, 7), dpi=300, sharex=False)
        elif num_rows == 2:
            fig, axs = plt.subplots(ncols=4, nrows=num_rows, figsize=(12, 9), dpi=300, sharex=False)
        elif num_rows == 3:
            fig, axs = plt.subplots(ncols=4, nrows=num_rows, figsize=(12, 11), dpi=300, sharex=False)
        else:
            fig, axs = plt.subplots(ncols=4, nrows=num_rows, figsize=(12, 13), dpi=300, sharex=False)

        col: int = 0
        row: int = 0

        # Extract all available experiment names
        experiment_names: [] = [experiment_name for experiment_name in vae_r2_scores.keys()]

        for experiment_name in experiment_names:
            vae_r2_score = vae_r2_scores[experiment_name]
            ae_r2_score = ae_r2_scores[experiment_name]
            en_r2_score = en_r2_scores[experiment_name]
            me_vae_r2_score = me_vae_r2_scores[experiment_name]

            sns.barplot(x='Marker', y='Score', data=en_r2_score, ax=axs[row, col])
            axs[row, col].set_title(f"{experiment_name} EN")
            axs[row, col].set_xticklabels(axs[row, col].get_xticklabels(), rotation=90)
            axs[row, col].set_ylim(0, 1)
            col += 1

            sns.barplot(x='Marker', y='Score', data=ae_r2_score, ax=axs[row, col])
            axs[row, col].set_title(f"{experiment_name} AE")
            axs[row, col].set_xticklabels(axs[row, col].get_xticklabels(), rotation=90)
            axs[row, col].set_ylim(0, 1)
            col += 1

            sns.barplot(x='Marker', y='Score', data=vae_r2_score, ax=axs[row, col])
            axs[row, col].set_title(f"{experiment_name} VAE")
            axs[row, col].set_xticklabels(axs[row, col].get_xticklabels(), rotation=90)
            axs[row, col].set_ylim(0, 1)
            col += 1

            sns.barplot(x='Marker', y='Score', data=me_vae_r2_score, ax=axs[row, col])
            axs[row, col].set_title(f"{experiment_name} ME VAE")
            axs[row, col].set_xticklabels(axs[row, col].get_xticklabels(), rotation=90)
            axs[row, col].set_ylim(0, 1)
            col += 1

            row += 1
            col = 0

        plt.ylim(0, 1)
        plt.legend(title='Model', loc='upper left', labels=['VAE', 'AE', 'EN'])
        fig.tight_layout()
        save_path = Path(self.__base_path, f"{file_name}.png")
        plt.savefig(save_path)
        mlflow.log_artifact(str(save_path))
        plt.close()

    def r2_scores_distribution(self, r2_scores: dict, file_name: str, mlflow_directory: str = None):
        """
        Plots a graph using a dictionary
        @param r2_scores: The scores to plot. Keys are sub plot names, values are scores
        @param file_name: The file name
        @param mlflow_directory: The mlflow directory
        @return:
        """
        num_rows = 1
        if len(r2_scores.items()) > 3:
            num_rows = float(len(r2_scores.items()) / 3)
            if not num_rows.is_integer():
                num_rows += 1

            num_rows = int(num_rows)

        n_cols = 3

        # Adjust columns based on items
        if num_rows == 1:
            fig, axs = plt.subplots(ncols=len(r2_scores.keys()), nrows=num_rows, figsize=(12, 7), dpi=300, sharex=False)
        elif num_rows == 2:
            fig, axs = plt.subplots(ncols=3, nrows=num_rows, figsize=(12, 9), dpi=300, sharex=False)
        elif num_rows == 3:
            fig, axs = plt.subplots(ncols=3, nrows=num_rows, figsize=(12, 11), dpi=300, sharex=False)
        else:
            fig, axs = plt.subplots(ncols=3, nrows=num_rows, figsize=(12, 13), dpi=300, sharex=False)

        col: int = 0
        row: int = 0

        if num_rows == 1:
            for experiment_name, r2_score in r2_scores.items():

                if len(r2_scores.items()) == 1:
                    sns.barplot(x='Marker', y='Score', data=r2_score, ax=axs)
                    axs.set_title(experiment_name)
                    axs.set_xticklabels(axs.get_xticklabels(), rotation=90)
                    axs[row, col].set_ylim(0, 1)
                else:
                    sns.boxplot(data=r2_score, ax=axs[col])
                    axs[col].set_title(experiment_name)
                    axs[col].set_xticklabels(axs[col].get_xticklabels(), rotation=90)
                    axs[col].set_ylim(0, 1)
                    col += 1

        else:
            for experiment_name, r2_score in r2_scores.items():
                sns.boxplot(data=r2_score, ax=axs[row, col])
                axs[row, col].set_title(experiment_name)
                axs[row, col].set_xticklabels(axs[row, col].get_xticklabels(), rotation=90)
                axs[row, col].set_ylim(0, 1)
                col += 1

                if col == n_cols:
                    row += 1
                    col = 0

        fig.tight_layout()
        save_path = Path(self.__base_path, f"{file_name}.png")
        plt.savefig(save_path)

        if mlflow_directory is not None:
            mlflow.log_artifact(str(save_path), mlflow_directory)
        else:
            mlflow.log_artifact(str(save_path))
        plt.close()

    def plot_weights_distribution(self, weights: pd.DataFrame, layer: str, prefix: str = None):
        fig, ax1 = plt.subplots(ncols=1, figsize=(28, 20), dpi=300, sharex=False)
        df = pd.DataFrame(columns=["Weights", "Markers"])
        for column, weights in weights.iteritems():
            for weight in weights.values:
                df = df.append({
                    "Marker": column,
                    "Weights": weight
                }, ignore_index=True)

        sns.displot(df, x="Weights", hue="Marker", legend=True)
        fig.tight_layout()
        plt.legend(loc='lower center')

        if prefix is not None:
            save_path = Path(self.__base_path, f"{prefix}_{layer}_weights_distribution.png")
        else:
            save_path = Path(self.__base_path, f"{layer}_weights_distribution.png")
        plt.savefig(save_path)

        mlflow.log_artifact(str(save_path))
        plt.close()

    def cross_fold_evaluation(self, evaluation_data: list, value_to_display: str, file_name: str, mlflow_folder: str):
        """
        Plots a barplot given the evaluation data of the cross fold validation.
        @param evaluation_data: The data of each run
        @param value_to_display: E.g. reconstruction_loss, or kl_loss. Uses the column to display the distribution
        @param file_name: The file name for the results
        @param mlflow_folder: Where to store the resulting file?
        @return:
        """

        df = pd.DataFrame(evaluation_data)
        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(9, 12), dpi=300, sharex=False)
        sns.boxplot(x=df["amount_of_layers"], y=df[f"{value_to_display}"])
        ax.set_title("Cross Fold model performance")

        save_path = Path(self.__base_path, f"{file_name}.png")
        plt.savefig(save_path)
        mlflow.log_artifact(str(save_path), mlflow_folder)
        plt.close()

    def plot_model_architecture(self, model, file_name: str, mlflow_folder: str = None):

        save_path = Path(self.__base_path, f"{file_name}.png")

        plot_model(model, save_path)

        if mlflow_folder is None:
            mlflow.log_artifact(str(save_path))
        else:
            mlflow.log_artifact(str(save_path), mlflow_folder)

    def plot_correlation(self, data_set: pd.DataFrame, file_name: str, mlflow_folder: str = None):
        """
        Plots the correlation for the given dataset
        @param data_set:
        @param file_name:
        @param mlflow_folder:
        @return:
        """
        correlations: pd.DataFrame = data_set.corr(method='spearman')

        mask = np.zeros_like(correlations)
        mask[np.triu_indices_from(mask)] = True
        with sns.axes_style("white"):
            f, ax = plt.subplots(figsize=(7, 5))

        ax = sns.heatmap(correlations, mask=mask, vmax=1, square=True)
        ax.set_title("Correlation")
        fig = ax.get_figure()
        fig.tight_layout()

        save_path = Path(self.__base_path, f"{file_name}.png")
        plt.savefig(save_path)
        if mlflow_folder is None:
            mlflow.log_artifact(str(save_path))
        else:
            mlflow.log_artifact(str(save_path), mlflow_folder)

        plt.close()
