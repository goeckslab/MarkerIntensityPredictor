import pandas as pd
from services.data_loader import DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso, LinearRegression


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
        inputs, self.__markers = DataLoader.get_data(self.__input_file)
        self.__inputs = np.array(inputs)
        self.__X_dev, self.__X_val = train_test_split(inputs, test_size=0.05, random_state=1, shuffle=True)
        self.__X_train, self.__X_test = train_test_split(self.__X_dev, test_size=0.25, random_state=1)

    def train_predict(self):
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

            ls_model = Lasso(alpha=1)
            ls_model.fit(X_train, y_train)
            self.prediction_scores = self.prediction_scores.append({
                "Marker": marker,
                "Score": ls_model.score(X_test, y_test),
                "Model": "Lasso"
            }, ignore_index=True)

            lr_model = LinearRegression()
            lr_model.fit(X_train, y_train)
            self.prediction_scores = self.prediction_scores.append({
                "Marker": marker,
                "Score": lr_model.score(X_test, y_test),
                "Model": "LR"
            }, ignore_index=True)

        print(self.prediction_scores)
        self.prediction_scores.to_csv("prediction_scores.csv")
