import pandas as pd
from sklearn.metrics import r2_score
from library.preprocessing.replacements import Replacer
from library.predictions.predictions import Predictions
from library.preprocessing.split import SplitHandler


class MEVAEImputation:
    morph_data: list = ["Area", "MajorAxisLength", "MinorAxisLength", "Solidity", "Extent"]

    @staticmethod
    def impute_data_by_feature(model, feature_to_impute: str, marker_data: pd.DataFrame, morph_data: pd.DataFrame,
                               ground_truth_marker_data: pd.DataFrame, ground_truth_morph_data: pd.DataFrame,
                               percentage: float,
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
            working_morph_data, indexes = Replacer.replace_values_by_feature(data=working_morph_data,
                                                                             feature_to_replace=feature_to_impute,
                                                                             percentage=percentage)
        else:
            working_marker_data, indexes = Replacer.replace_values_by_feature(data=working_marker_data,
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

    @staticmethod
    def impute_data_by_cell(model, marker_data: pd.DataFrame, morph_data: pd.DataFrame,
                            ground_truth_marker_data: pd.DataFrame, ground_truth_morph_data: pd.DataFrame,
                            percentage: float, iter_steps: int, features: list):
        """

        @param model:
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
        replaced_marker_data = marker_data.copy()
        replaced_morph_data = morph_data.copy()

        frames = [replaced_marker_data, replaced_morph_data]
        # Merge data together for replacements
        replaced_data = pd.concat(frames, axis=1)

        # : Contains the index and feature which where replaced in the particular row
        replaced_data, index_replacements = Replacer.replace_values_by_cell(data=replaced_data,
                                                                            features=features,
                                                                            percentage=percentage)

        # Split data back into marker and morph data
        replaced_marker_data, replaced_morph_data = SplitHandler.split_dataset_into_markers_and_morph_features(
            data_set=replaced_data.copy())

        imputed_marker_data: pd.DataFrame = replaced_marker_data.copy()
        imputed_morph_data: pd.DataFrame = replaced_morph_data.copy()

        # Iterate to impute
        for i in range(iter_steps):
            # Predict embeddings and mean
            mean, log_var, z = model.encoder.predict([imputed_marker_data, imputed_morph_data])

            # Create reconstructed date
            reconstructed_data = pd.DataFrame(columns=features, data=model.decoder.predict(mean))

            # Combined imputation datasets to replace reconstructed data
            frames = [imputed_marker_data, imputed_morph_data]
            imputed_data = pd.concat(frames, axis=1)
            for index, row in reconstructed_data.iterrows():
                replaced_features: list = index_replacements[index]
                # Update only replaced data
                for replaced_feature in replaced_features:
                    imputed_data.at[index, replaced_feature] = reconstructed_data.at[index, replaced_feature]

            # Split imputed data again
            imputed_marker_data, imputed_morph_data = SplitHandler.split_dataset_into_markers_and_morph_features(
                data_set=imputed_data)

        # Reconstruct unmodified test data
        encoded_data, reconstructed_data = Predictions.encode_decode_me_vae_data(encoder=model.encoder,
                                                                                 decoder=model.decoder,
                                                                                 data=[ground_truth_marker_data,
                                                                                       ground_truth_morph_data],
                                                                                 features=features,
                                                                                 use_mlflow=False)

        frames = [imputed_marker_data, imputed_morph_data]
        # Create full dataset with imputed data
        imputed_data = pd.concat(frames, axis=1)

        imputed_r2_scores: pd.DataFrame = pd.DataFrame()
        reconstructed_r2_scores: pd.DataFrame = pd.DataFrame()
        replaced_r2_scores: pd.DataFrame = pd.DataFrame()

        # Create complete ground truth dataset
        ground_truth_frames = [ground_truth_marker_data, ground_truth_morph_data]
        ground_truth_data: pd.DataFrame = pd.concat(ground_truth_frames, axis=1)

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

        # Return imputed, reconstructed and working data
        return imputed_r2_scores, reconstructed_r2_scores, replaced_r2_scores
