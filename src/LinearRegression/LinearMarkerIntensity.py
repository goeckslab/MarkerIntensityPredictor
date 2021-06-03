import pandas as pd
from services.data_loader import DataLoader
from sklearn.linear_model import Ridge, Lasso, LinearRegression, ElasticNet
import seaborn as sns
import re
from entities.data_storage import DataStorage
import os
from pathlib import Path
import matplotlib.pyplot as plt
import logging

sns.set_theme(style="darkgrid")


class LinearMarkerIntensity:
    train_file: str
    test_file: str
    train_file_name: str
    test_file_name: str
    train_data: DataStorage
    test_data: DataStorage

    coefficients = pd.DataFrame()
    prediction_scores = pd.DataFrame(columns=["Marker", "Score", "Model"])

    __args = list

    def __init__(self, train_file, args, validation_file):
        self.__args = args

        # Multi file
        if self.__args.multi:
            self.test_file = train_file
            self.test_file_name = os.path.splitext(self.test_file)[0].split('/')[1]
            self.train_file = None
            self.train_file_name = None

        # Validation file
        elif self.__args.validation is not None:
            self.train_file = train_file
            self.test_file = validation_file
            self.train_file_name = os.path.splitext(self.train_file)[0].split('/')[1]
            self.test_file_name = os.path.splitext(self.test_file)[0].split('/')[1]

        # Single File
        else:
            self.train_file = train_file
            self.train_file_name = os.path.splitext(self.train_file)[0].split('/')[1]
            self.test_file = None
            self.test_file_name = None

    def load(self):
        # Multi file
        if self.__args.multi:
            inputs, markers = DataLoader.get_data(self.test_file)
            self.test_data = DataStorage(inputs, markers)

            inputs, markers = DataLoader.merge_files(self.test_file)
            self.train_data = DataStorage(inputs, markers)

        # Validation file
        elif self.__args.validation is not None:
            inputs, markers = DataLoader.get_data(self.train_file)
            self.train_data = DataStorage(inputs, markers)
            inputs, markers = DataLoader.get_data(self.test_file)
            self.test_data = DataStorage(inputs, markers)

        # Single File
        else:
            inputs, markers = DataLoader.get_data(self.train_file)
            self.train_data = DataStorage(inputs, markers)

    def train_predict(self):
        self.coefficients = pd.DataFrame(columns=self.train_data.X_train.columns)
        self.coefficients["Model"] = ""

        for marker in self.train_data.X_train.columns:
            # Copy df to not change
            train_copy = self.train_data.X_train.copy()

            if self.test_file is not None:
                test_copy = self.test_data.X_test.copy()
            else:
                test_copy = self.train_data.X_test.copy()

            if marker == "ERK1_1":
                del train_copy["ERK1_2"]
                del test_copy["ERK1_2"]

            if marker == "ERK1_2":
                del train_copy["ERK1_1"]
                del test_copy["ERK1_1"]

            # Create y and X
            y_train = train_copy[f"{marker}"]
            del train_copy[f"{marker}"]
            X_train = train_copy

            y_test = test_copy[f"{marker}"]
            del test_copy[f"{marker}"]
            X_test = test_copy

            self.__predict("Ridge", Ridge(alpha=10), X_train, y_train, X_test, y_test, marker)
            self.__predict("LR", LinearRegression(), X_train, y_train, X_test, y_test, marker)
            self.__predict("Lasso", Lasso(alpha=0.5), X_train, y_train, X_test, y_test, marker)
            self.__predict("EN", ElasticNet(alpha=0.5), X_train, y_train, X_test, y_test, marker)

        self.prediction_scores["Marker"] = [re.sub("_nucleiMasks", "", x) for x in self.prediction_scores["Marker"]]

    def write_csv(self):
        """
        Creates csv files for each important csv
        :return:
        """
        if self.test_file is None:
            self.coefficients.to_csv(Path(f"results/lr/{self.train_file_name}_coefficients.csv"))
            self.prediction_scores.to_csv(Path(f"results/lr/{self.train_file_name}_prediction_scores.csv"))

        elif self.train_file is None:
            self.coefficients.to_csv(Path(f"results/lr/{self.test_file_name}_multi_coefficients.csv"))
            self.prediction_scores.to_csv(Path(f"results/lr/{self.test_file_name}_multi_prediction_scores.csv"))
        else:
            self.coefficients.to_csv(Path(f"results/lr/{self.train_file_name}_{self.test_file_name}_coefficients.csv"))
            self.prediction_scores.to_csv(
                Path(f"results/lr/{self.train_file_name}_{self.test_file_name}_prediction_scores.csv"))

    def create_plots(self):
        """
        Creates all plots associated to the model
        :return:
        """
        self.__create_r2_accuracy_plot()
        self.__create_coef_heatmap_plot()
        self.__create_intensity_heatmap_plot()

    def __create_coefficient_df(self, train, model, marker):
        """
        Creates a dataset which contains the coefficents for each marker
        :param train:
        :param model:
        :return:
        """
        temp = pd.DataFrame(zip(train.columns, model.coef_))
        temp.rename(columns={0: 'Marker', 1: marker}, inplace=True)

        temp = temp.T
        new_header = temp.iloc[0]
        temp = temp[1:]
        temp.columns = new_header
        return temp

    def __create_coef_heatmap_plot(self):
        """
        Creates a heatmap for the coefficients
        """

        for model in self.coefficients["Model"].unique():
            df = self.coefficients[self.coefficients["Model"] == model].copy()

            del df["Model"]
            df.fillna(-1, inplace=True)
            # df[(df < 0.001) | (df > 0.6)] = None
            fig, ax = plt.subplots(figsize=(20, 20))  # Sample figsize in inches
            ax = sns.heatmap(df, linewidths=1, vmin=0, vmax=0.6, cmap="YlGnBu", ax=ax)
            ax.set_title(model)
            fig = ax.get_figure()
            fig.savefig(Path(f"results/lr/{model}_coef_heatmap.png"), bbox_inches='tight')
            plt.close()

    def __create_intensity_heatmap_plot(self):
        fig, ax = plt.subplots(figsize=(30, 30), dpi=300)  # Sample figsize in inches
        sns.heatmap(self.train_data.X_train, xticklabels=self.train_data.markers)
        ax.set_title("Marker intensities")
        fig = ax.get_figure()
        fig.tight_layout()
        fig.savefig(Path(f"results/lr/marker_heatmap.png"), bbox_inches='tight')
        plt.close()

    def __create_r2_accuracy_plot(self):
        """
        Creates a bar plot showing the accuracy of the model for each marker
        :return:
        """
        ax = sns.catplot(
            data=self.prediction_scores, kind="bar",
            x="Score", y="Marker", hue="Model",
            ci="sd", palette="dark", alpha=.6, height=6
        )
        ax.despine(left=True)
        ax.set_axis_labels("R2 Score", "Marker")
        ax.set(xlim=(0, 1))

        if self.test_file is None:
            # ax.fig.suptitle("Single file")
            plt.title("Single File", y=1.02)
            ax.legend.set_title("Model")
            ax.savefig(Path(f"results/lr/{self.train_file_name}_score_predictions.png"))
        elif self.train_file is None:
            plt.title("Multi Files", y=1.02)
            ax.legend.set_title("Model")
            ax.savefig(Path(f"results/lr/{self.test_file_name}_multi_score_predictions.png"))
        else:
            plt.title("Train Test File", y=1.02)
            ax.legend.set_title("Model")
            ax.savefig(Path(f"results/lr/{self.train_file_name}_{self.test_file_name}_score_predictions.png"))

        plt.close()

    def __predict(self, name: str, model, X_train, y_train, X_test, y_test, marker):
        model.fit(X_train, y_train)
        self.prediction_scores = self.prediction_scores.append({
            "Marker": marker,
            "Score": model.score(X_test, y_test),
            "Model": name
        }, ignore_index=True)

        self.coefficients = self.coefficients.append(self.__create_coefficient_df(X_train, model, marker))
        self.coefficients["Model"].fillna(name, inplace=True)
