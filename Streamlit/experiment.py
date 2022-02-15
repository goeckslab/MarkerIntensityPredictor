import pandas as pd
import numpy as np
import mlflow

class Experiment:
    name: str
    markers: list = None
    embeddings: pd.DataFrame = None

    def __init__(self, experiment_name: str):
        self.name = experiment_name
        self.__load_files()

    def __load_files(self):
        for file in self.files:
            if file.name == "markers.csv":
                self.markers = pd.read_csv(file)
            elif file.name == "encoded_data.csv":
                self.embeddings = pd.read_csv(file)
