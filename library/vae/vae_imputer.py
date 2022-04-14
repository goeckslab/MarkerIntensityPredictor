from sklearn.metrics import r2_score
from library.preprocessing.replacements import Replacer
import pandas as pd
from library.predictions.predictions import Predictions


class VAEImputation:

    @staticmethod
    def impute_data(model, iter_steps: int, ground_truth_data: pd.DataFrame, feature_to_impute: str, percentage: float,
                    features: list):
        """
        VAE imputation
        @param model:
        @param iter_steps:
        @param ground_truth_data:
        @param feature_to_impute:
        @param percentage:
        @param features:
        @return:  Returns imputed values, reconstructed then replaced
        """
        # Make a fresh copy, to start with the ground truth data
        working_data = ground_truth_data.copy()

        working_data, indexes = Replacer.replace_values(data=working_data, feature_to_replace=feature_to_impute,
                                                        percentage=percentage)

        imputed_data: pd.DataFrame = working_data.iloc[indexes].copy()

        # Iterate to impute
        for i in range(iter_steps):
            # Predict embeddings and mean
            mean, log_var, z = model.encoder.predict(imputed_data)

            # Create reconstructed date
            reconstructed_data = pd.DataFrame(columns=features, data=model.decoder.predict(mean))

            # Overwrite imputed data with reconstructed data
            imputed_data = reconstructed_data



        # Reconstruct unmodified test data
        encoded_data, reconstructed_data = Predictions.encode_decode_vae_data(encoder=model.encoder,
                                                                              decoder=model.decoder,
                                                                              data=ground_truth_data,
                                                                              features=features, use_mlflow=False)

        reconstructed_r2_scores = r2_score(ground_truth_data[feature_to_impute].iloc[indexes],
                                           reconstructed_data[feature_to_impute].iloc[indexes])

        imputed_r2_scores = r2_score(ground_truth_data[feature_to_impute].iloc[indexes],
                                     imputed_data[feature_to_impute])

        replaced_r2_scores = r2_score(ground_truth_data[feature_to_impute].iloc[indexes],
                                      working_data[feature_to_impute].iloc[indexes])

        return imputed_r2_scores, reconstructed_r2_scores, replaced_r2_scores
