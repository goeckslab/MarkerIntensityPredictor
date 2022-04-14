import pandas as pd
from sklearn.metrics import r2_score
from library.preprocessing.replacements import Replacer
from library.predictions.predictions import Predictions


class MEVAEImputation:
    morph_data: list = ["Area", "MajorAxisLength", "MinorAxisLength", "Solidity", "Extent"]

    @staticmethod
    def impute_data(model, feature_to_impute: str, marker_data: pd.DataFrame, morph_data: pd.DataFrame,
                    ground_truth_marker_data: pd.DataFrame, ground_truth_morph_data: pd.DataFrame, percentage: float,
                    iter_steps: int, features: list):
        """

        @param model:
        @param feature_to_impute:
        @param marker_data:
        @param morph_data:
        @param ground_truth_marker_data:
        @param ground_truth_morph_data:
        @param percentage:
        @param iter_steps:
        @param features:
        @return: Returns a tuple of imputed scores, reconstructed and replaced
        """

        # Make a fresh copy, to start with the ground truth data
        working_marker_data = marker_data.copy()
        working_morph_data = morph_data.copy()

        # Depending on marker and morph data replace only one feature
        if feature_to_impute in morph_data:
            working_morph_data, indexes = Replacer.replace_values(data=working_morph_data,
                                                                  feature_to_replace=feature_to_impute,
                                                                  percentage=percentage)
        else:
            working_marker_data, indexes = Replacer.replace_values(data=working_marker_data,
                                                                   feature_to_replace=feature_to_impute,
                                                                   percentage=percentage)

        imputed_marker_data: pd.DataFrame = working_marker_data.iloc[indexes].copy()
        imputed_morph_data: pd.DataFrame = working_morph_data.iloc[indexes].copy()

        # Iterate to impute
        for i in range(iter_steps):
            # Predict embeddings and mean
            mean, log_var, z = model.encoder.predict([imputed_marker_data, imputed_morph_data])

            # Create reconstructed date
            reconstructed_data = pd.DataFrame(columns=features, data=model.decoder.predict(mean))

            # Select correct data depending on whether is a morph feature or marker
            if feature_to_impute in morph_data:
                imputed_morph_data = reconstructed_data[
                    ["Area", "MajorAxisLength", "MinorAxisLength", "Solidity", "Extent"]]

            else:
                imputed_marker_data = reconstructed_data.drop(
                    ["Area", "MajorAxisLength", "MinorAxisLength", "Solidity", "Extent"],
                    axis=1)

        # Reconstruct unmodified test data
        encoded_data, reconstructed_data = Predictions.encode_decode_me_vae_data(encoder=model.encoder,
                                                                                 decoder=model.decoder,
                                                                                 data=[ground_truth_marker_data,
                                                                                       ground_truth_morph_data],
                                                                                 features=features,
                                                                                 use_mlflow=False)

        if feature_to_impute in morph_data:
            reconstructed_r2_scores = r2_score(ground_truth_morph_data[feature_to_impute].iloc[indexes],
                                               reconstructed_data[feature_to_impute].iloc[indexes])

            imputed_r2_scores = r2_score(ground_truth_morph_data[feature_to_impute].iloc[indexes],
                                         imputed_morph_data[feature_to_impute])

            replaced_r2_scores = r2_score(ground_truth_morph_data[feature_to_impute].iloc[indexes],
                                          working_morph_data[feature_to_impute].iloc[indexes])
        else:
            reconstructed_r2_scores = r2_score(ground_truth_marker_data[feature_to_impute].iloc[indexes],
                                               reconstructed_data[feature_to_impute].iloc[indexes])

            imputed_r2_scores = r2_score(ground_truth_marker_data[feature_to_impute].iloc[indexes],
                                         imputed_marker_data[feature_to_impute])

            replaced_r2_scores = r2_score(ground_truth_marker_data[feature_to_impute].iloc[indexes],
                                          working_marker_data[feature_to_impute].iloc[indexes])

        return imputed_r2_scores, reconstructed_r2_scores, replaced_r2_scores
