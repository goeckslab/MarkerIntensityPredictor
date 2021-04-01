import pandas as pd
from services.data_loader import DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso, LinearRegression, ElasticNet
import seaborn as sns
import re
from services.normalization import Normalizer


class LinearMarkerIntensity:
    __input_file: str
    __inputs = pd.DataFrame()
    __model: str
    __markers: list
    __X_dev = pd.DataFrame()
    __X_val = pd.DataFrame()
    __X_train = pd.DataFrame()
    __X_test = pd.DataFrame()

    coefficients = pd.DataFrame()
    prediction_scores = pd.DataFrame(columns=["Marker", "Score", "Model"])

    def __init__(self, input_file, model='Ridge'):
        self.__input_file = input_file
        self.__model = model
        self.__inputs = pd.DataFrame()

    def load(self):
        self.__inputs, self.__markers = DataLoader.get_data(self.__input_file)
        self.__inputs = np.array(self.__inputs)
        self.__X_dev, self.__X_val = train_test_split(self.__inputs, test_size=0.05, random_state=1, shuffle=True)
        self.__X_train, self.__X_test = train_test_split(self.__X_dev, test_size=0.25, random_state=1)

        init_inputs = self.__inputs
        init_X_train = self.__X_train
        init_X_test = self.__X_test
        init_X_val = self.__X_val

        self.__inputs = pd.DataFrame(columns=self.__markers, data=Normalizer.normalize(self.__inputs))
        self.__X_train = pd.DataFrame(columns=self.__markers, data=Normalizer.normalize(self.__X_train))
        self.__X_test = pd.DataFrame(columns=self.__markers, data=Normalizer.normalize(self.__X_test))
        self.__X_val = pd.DataFrame(columns=self.__markers, data=Normalizer.normalize(self.__X_val))

    def train_predict(self):
        self.coefficients = pd.DataFrame(columns=self.__X_train.columns)
        self.coefficients["Model"] = ""
        for marker in self.__X_train.columns:
            # Copy df to not change
            train_copy = self.__X_train.copy()
            test_copy = self.__X_test.copy()

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

            self.__predict("Ridge", Ridge(alpha=1), X_train, y_train, X_test, y_test, marker)
            self.__predict("LR", LinearRegression(), X_train, y_train, X_test, y_test, marker)
            self.__predict("Lasso", Lasso(alpha=0.1), X_train, y_train, X_test, y_test, marker)
            self.__predict("EN", ElasticNet(alpha=0.1), X_train, y_train, X_test, y_test, marker)

        self.prediction_scores["Marker"] = [re.sub("_nucleiMasks", "", x) for x in self.prediction_scores["Marker"]]

    def write_csv(self):
        """
        Creates csv files for each important csv
        :return:
        """
        self.coefficients.to_csv("coefficients.csv")
        self.prediction_scores.to_csv("prediction_scores.csv")

    def create_plots(self):
        """
        Creates all plots associated to the model
        :return:
        """
        self.__create_r2_accuracy_plot()

    def __create_coefficent_df(self, train, model, marker):
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

    def __create_r2_accuracy_plot(self):
        """
        Creates a bar plot showing the accuracy of the model for each marker
        :return:
        """
        g = sns.catplot(
            data=self.prediction_scores, kind="bar",
            x="Score", y="Marker", hue="Model",
            ci="sd", palette="dark", alpha=.6, height=6
        )
        g.despine(left=True)
        g.set_axis_labels("R2 Score", "Marker")
        g.legend.set_title("")
        # g.set_xticklabels(rotation=90)
        g.savefig("score_predictions.png")

    def __predict(self, name: str, model, X_train, y_train, X_test, y_test, marker):

        model.fit(X_train, y_train)
        self.prediction_scores = self.prediction_scores.append({
            "Marker": marker,
            "Score": model.score(X_test, y_test),
            "Model": name
        }, ignore_index=True)

        self.coefficients = self.coefficients.append(self.__create_coefficent_df(X_train, model, marker))
        self.coefficients["Model"].fillna(name, inplace=True)
