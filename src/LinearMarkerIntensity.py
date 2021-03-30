import pandas as pd
from services.data_loader import DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso, LinearRegression
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
        coef = pd.DataFrame(columns=self.__X_train.columns)
        coef["Model"] = ""
        for marker in self.__X_train.columns:
            # Copy df to not change
            train_copy = self.__X_train.copy()

            # Create y and X
            y_train = train_copy[f"{marker}"]
            del train_copy[f"{marker}"]
            X_train = train_copy

            test_copy = self.__X_test.copy()
            y_test = test_copy[f"{marker}"]
            del test_copy[f"{marker}"]
            X_test = test_copy

            model = Ridge(alpha=1)
            model.fit(X_train, y_train)
            self.prediction_scores = self.prediction_scores.append({
                "Marker": marker,
                "Score": model.score(X_test, y_test),
                "Model": "Ridge"
            }, ignore_index=True)

            coef = coef.append(self.__create_coefficent_df(X_train, model))
            coef["Model"].fillna("Ridge", inplace=True)

            model = Lasso(alpha=1)
            model.fit(X_train, y_train)
            self.prediction_scores = self.prediction_scores.append({
                "Marker": marker,
                "Score": model.score(X_test, y_test),
                "Model": "Lasso"
            }, ignore_index=True)

            coef = coef.append(self.__create_coefficent_df(X_train, model))
            coef["Model"].fillna("Lasso",  inplace=True)

            model = LinearRegression()
            model.fit(X_train, y_train)
            self.prediction_scores = self.prediction_scores.append({
                "Marker": marker,
                "Score": model.score(X_test, y_test),
                "Model": "LR"
            }, ignore_index=True)

            coef = coef.append(self.__create_coefficent_df(X_train, model))
            coef["Model"].fillna("LR",  inplace=True)


        self.prediction_scores["Marker"] = [re.sub("_nucleiMasks", "", x) for x in self.prediction_scores["Marker"]]
        coef.to_csv("coef.csv")
        self.prediction_scores.to_csv("prediction_scores.csv")
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

    def __create_coefficent_df(self, train, model):
        temp = pd.DataFrame(zip(train.columns, model.coef_))
        temp.rename(columns={0: 'Marker', 1: "Coef"}, inplace=True)

        temp = temp.T
        new_header = temp.iloc[0]
        temp = temp[1:]
        temp.columns = new_header
        return temp