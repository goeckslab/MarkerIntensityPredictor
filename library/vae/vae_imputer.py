from sklearn.metrics import r2_score
from library.preprocessing.replacements import Replacer
import pandas as pd
from library.predictions.predictions import Predictions


class VAEImputation:

    @staticmethod
    def impute_data_by_feature(model, iter_steps: int, ground_truth_data: pd.DataFrame, feature_to_impute: str,
                               percentage: float, features: list):
        """
        VAE imputation
        @param model:
        @param iter_steps:
        @param ground_truth_data:
        @param feature_to_impute:
        @param percentage:
        @param features:
        @return: Returns the r2 scores for imputed values, reconstructed and replaced values
        """
        # Make a fresh copy, to start with the ground truth data
        working_data = ground_truth_data.copy()

        working_data, indexes = Replacer.replace_values_by_feature(data=working_data,
                                                                   feature_to_replace=feature_to_impute,
                                                                   percentage=percentage)

        imputed_data: pd.DataFrame = working_data.iloc[indexes].copy()

        # Iterate to impute
        for i in range(iter_steps):
            # Predict embeddings and mean
            mean, log_var, z = model.encoder.predict(imputed_data)

            # Create reconstructed date
            reconstructed_data = pd.DataFrame(columns=features, data=model.decoder.predict(mean))

            # Overwrite imputed data with reconstructed data
            imputed_data[feature_to_impute] = reconstructed_data[feature_to_impute].values

        # Reconstruct unmodified test data
        encoded_data, reconstructed_data = Predictions.encode_decode_vae_data(encoder=model.encoder,
                                                                              decoder=model.decoder,
                                                                              data=ground_truth_data,
                                                                              features=features, use_mlflow=False)

        reconstructed_r2_score = r2_score(ground_truth_data[feature_to_impute].iloc[indexes],
                                          reconstructed_data[feature_to_impute].iloc[indexes])

        imputed_r2_score = r2_score(ground_truth_data[feature_to_impute].iloc[indexes],
                                    imputed_data[feature_to_impute])

        replaced_r2_score = r2_score(ground_truth_data[feature_to_impute].iloc[indexes],
                                     working_data[feature_to_impute].iloc[indexes])

        return imputed_r2_score, reconstructed_r2_score, replaced_r2_score

    @staticmethod
    def impute_data_by_cell(model, iter_steps: int, ground_truth_data: pd.DataFrame, percentage: float,
                            features: list):
        """
        VAE imputation
        @param model:
        @param iter_steps:
        @param ground_truth_data:
        @param percentage:
        @param features:
        @return:  Returns imputed values, reconstructed then replaced
        """
        # Make a fresh copy, to start with the ground truth data
        replaced_data = ground_truth_data.copy()

        replaced_data, index_replacements = Replacer.replace_values_by_cell(data=replaced_data, features=features,
                                                                            percentage=percentage)

        imputed_data: pd.DataFrame = replaced_data.copy()

        # Iterate to impute
        for i in range(iter_steps):
            # Predict embeddings and mean
            mean, log_var, z = model.encoder.predict(imputed_data)

            # Create reconstructed date
            reconstructed_data = pd.DataFrame(columns=features, data=model.decoder.predict(mean))

            # Overwrite imputed data with reconstructed data
            for index, row in reconstructed_data.iterrows():
                replaced_features: list = index_replacements[index]
                # Update only replaced data
                for replaced_feature in replaced_features:
                    imputed_data.at[index, replaced_feature] = reconstructed_data.at[index, replaced_feature]
            imputed_data = reconstructed_data

        # Reconstruct unmodified test data
        encoded_data, reconstructed_data = Predictions.encode_decode_vae_data(encoder=model.encoder,
                                                                              decoder=model.decoder,
                                                                              data=ground_truth_data,
                                                                              features=features, use_mlflow=False)

        imputed_r2_scores: pd.DataFrame = pd.DataFrame()
        reconstructed_r2_scores: pd.DataFrame = pd.DataFrame()
        replaced_r2_scores: pd.DataFrame = pd.DataFrame()

        for feature in features:
            # Store all cell indexes, to be able to select the correct cells later for r2 comparison
            cell_indexes_to_compare: list = []
            for key, replaced_features in index_replacements.items():
                if feature in replaced_features:
                    cell_indexes_to_compare.append(key)

            imputed_r2_scores = imputed_r2_scores.append({
                "Marker": feature,
                "Score": r2_score(ground_truth_data[feature].iloc[cell_indexes_to_compare],
                                  imputed_data[feature].iloc[cell_indexes_to_compare])
            }, ignore_index=True)

            reconstructed_r2_scores = reconstructed_r2_scores.append({
                "Marker": feature,
                "Score": r2_score(ground_truth_data[feature].iloc[cell_indexes_to_compare],
                                  reconstructed_data[feature].iloc[cell_indexes_to_compare])
            }, ignore_index=True)

            replaced_r2_scores = replaced_r2_scores.append({
                "Marker": feature,
                "Score": r2_score(ground_truth_data[feature].iloc[cell_indexes_to_compare],
                                  replaced_data[feature].iloc[cell_indexes_to_compare])
            }, ignore_index=True)

        return imputed_r2_scores, reconstructed_r2_scores, replaced_r2_scores
