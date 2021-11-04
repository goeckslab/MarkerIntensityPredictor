import pandas as pd
from Shared.data_loader import DataLoader
from sklearn.linear_model import Ridge, Lasso, LinearRegression, ElasticNet
import re
import numpy as np
import os
from pathlib import Path
from Shared.data import Data
from sklearn.preprocessing import StandardScaler


class LinearMarkerIntensity:
    train_file: str
    test_file: str
    train_file_name: str
    test_file_name: str
    train_data: Data = None
    test_data: Data = None

    coefficients = pd.DataFrame()
    r2scores = pd.DataFrame(columns=["Marker", "Score", "Model"])

    args = None
    results_folder = Path("results", "lr")

    def __init__(self, train_file, validation_file, args):
        self.args = args

        # Multi file
        if self.args.multi:
            self.test_file = train_file
            self.test_file_name = os.path.splitext(self.test_file)[0].split('/')[1]
            self.train_file = None
            self.train_file_name = None

        # Validation file
        elif self.args.validation is not None:
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
        if self.args.multi:
            inputs, markers = DataLoader.get_data(self.test_file, False)
            self.test_data = Data(inputs=inputs, markers=markers, normalize=self.normalize)

            inputs, markers = DataLoader.merge_files(self.test_file)
            self.train_data = Data(inputs=inputs, markers=markers, normalize=self.normalize)

        # Validation file
        elif self.args.validation is not None:
            inputs, markers = DataLoader.get_data(self.train_file, False)
            self.train_data = Data(inputs=inputs, markers=markers, normalize=self.normalize)
            inputs, markers = DataLoader.get_data(self.test_file, False)
            self.test_data = Data(inputs=inputs, markers=markers, normalize=self.normalize)

        # Single File
        else:
            inputs, markers = DataLoader.get_data(self.train_file, False)
            self.train_data = Data(inputs=inputs, markers=markers, normalize=self.normalize)

    def train_predict(self):
        self.coefficients = pd.DataFrame(columns=self.train_data.markers)
        self.coefficients["Model"] = ""

        for marker in self.train_data.markers:
            # Copy df to not change
            train_copy = pd.DataFrame(columns=self.train_data.markers, data=self.train_data.X_train.copy())

            if self.test_file is not None:
                test_copy = pd.DataFrame(columns=self.test_data.markers, data=self.test_data.X_test.copy())
            else:
                test_copy = pd.DataFrame(columns=self.train_data.markers, data=self.train_data.X_test.copy())

            if marker == "ERK1_1" and 'ERK1_2' in self.train_data.markers:
                del train_copy["ERK1_2"]
                del test_copy["ERK1_2"]

            if marker == "ERK1_2" and 'ERK1_1' in self.train_data.markers:
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
            self.__predict("Lasso", Lasso(alpha=0.1), X_train, y_train, X_test, y_test, marker)
            self.__predict("EN", ElasticNet(alpha=0.1), X_train, y_train, X_test, y_test, marker)

        self.r2scores["Marker"] = [re.sub("_nucleiMasks", "", x) for x in self.r2scores["Marker"]]

    def write_csv(self):
        """
        Creates csv files for each important csv
        :return:
        """

        self.coefficients.to_csv(
            Path(f"{self.results_folder}/coefficients.csv"))
        self.r2scores.to_csv(
            Path(f"results/lr/r2_scores.csv"), index=False)

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

    def __predict(self, name: str, model, X_train, y_train, X_test, y_test, marker):
        model.fit(X_train, y_train)
        self.r2scores = self.r2scores.append({
            "Marker": marker,
            "Score": model.score(X_test, y_test),
            "Model": name
        }, ignore_index=True)

        self.coefficients = self.coefficients.append(self.__create_coefficient_df(X_train, model, marker))
        self.coefficients["Model"].fillna(name, inplace=True)

    def normalize(self, data):
        # Input data contains some zeros which results in NaN (or Inf)
        # values when their log10 is computed. NaN (or Inf) are problematic
        # values for downstream analysis. Therefore, zeros are replaced by
        # a small value; see the following thread for related discussion.
        # https://www.researchgate.net/post/Log_transformation_of_values_that_include_0_zero_for_statistical_analyses2
        data[data == 0] = 1e-32
        data = np.log10(data)

        standard_scaler = StandardScaler()
        data = standard_scaler.fit_transform(data)
        data = data.clip(min=-5, max=5)

        # min_max_scaler = MinMaxScaler()
        # data = min_max_scaler.fit_transform(data)
        return data
