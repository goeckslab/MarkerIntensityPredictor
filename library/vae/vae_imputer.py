import pandas as pd
from tensorflow.keras.models import Model
from typing import Dict


class VAEImputer:

    def __init__(self, model: Model, index_replacements: Dict, replaced_data: pd.DataFrame, iterations: int,
                 features_to_impute: str):
        self._model = model
        self._iterations = iterations
        self._features_to_impute = features_to_impute
        self._replaced_data = replaced_data
        self._index_replacements = index_replacements
        self._imputed_data = pd.DataFrame()

    @property
    def imputed_data(self):
        if self._imputed_data.empty:
            print("Imputed data is empty. Please call impute() to create data.")
        return self._imputed_data

    @property
    def index_replacements(self):
        return self._index_replacements

    def impute(self):
        print("Imputing data...")
        self._imputed_data = self._replaced_data.copy()

        # Iterate to impute
        for i in range(self._iterations):
            # Predict embeddings and mean
            mean, log_var, z = self._model.encoder.predict(self._imputed_data)

            # Create reconstructed date
            reconstructed_data = pd.DataFrame(columns=self._replaced_data.columns,
                                              data=self._model.decoder.predict(mean))

            # Overwrite imputed data with reconstructed data
            for index, row in reconstructed_data.iterrows():
                replaced_features: list = self._index_replacements[index]
                # Update only replaced data
                for replaced_feature in replaced_features:
                    self._imputed_data.at[index, replaced_feature] = reconstructed_data.at[index, replaced_feature]
