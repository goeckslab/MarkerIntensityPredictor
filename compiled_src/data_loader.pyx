#cython: language_level=3
import pandas as pd
from pathlib import Path
import re
import os
from typing import Tuple, Optional


class DataLoader:

    @staticmethod
    def load_single_cell_data(file_name: str, keep_morph: bool = True, keep_spatial: bool = False,
                              return_df: bool = False):
        """
        Loads the marker data, given the provided information
        @param file_name: The file name to load from the os
        @param keep_morph: Keep morphological features or sort them out? Default to true
        @param keep_spatial: Keep spatial features or sort them out? Default to false
        @return:
        """
        columns_to_remove = ["CellID", "ERK1_2_nucleiMasks", "Orientation"]

        path = Path(file_name)
        cells = pd.read_csv(path, header=0)

        columns_to_keep: list = cells.copy().filter(regex="nucleiMasks$", axis=1).filter(regex="^(?!(DAPI|AF))",
                                                                                         axis=1).columns.tolist()
        columns_to_keep.remove('ERK1_2_nucleiMasks')

        if keep_morph:
            columns_to_keep.extend(["Area", "MajorAxisLength", "MinorAxisLength", "Solidity", "Extent"])

        if keep_spatial:
            columns_to_keep.extend(["X_centroid", "Y_centroid"])

        new_cells: pd.DataFrame = pd.DataFrame()

        # Add columns to new dataframe
        for column in columns_to_keep:
            new_cells[f"{column}"] = cells[f"{column}"]

        # Remove manually not required columns
        for column in columns_to_remove:
            if column in new_cells.columns:
                del new_cells[f"{column}"]

        features = new_cells.columns
        features = [re.sub("_nucleiMasks", "", x) for x in features]
        new_cells.columns = features

        for column in columns_to_keep:
            column = re.sub("_nucleiMasks", "", column)
            assert f"{column}" in features, f"{column} should be in features"

        for column in columns_to_remove:
            column = re.sub("_nucleiMasks", "", column)
            assert f"{column}" not in features, f"{column} should not be in features"

        if return_df:
            return pd.DataFrame(data=new_cells, columns=features)
        else:
            # return cells, markers
            return new_cells.iloc[:, :], features

    @staticmethod
    def load_r2_scores_for_model(load_path: Path, file_name: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Loads all r2scores for the given model and combines them in a dataset
        @param load_path: The path from where to load the data
        @param file_name: The file name to search for
        @return: Returns a tuple of dataframes. First dataframe contains the mean scores, the second contains all combined scores.
        """
        combined_r2_scores = pd.DataFrame()
        markers = []

        for p in load_path.rglob("*"):
            if p.name == file_name:
                df = pd.read_csv(p.absolute(), header=0)
                combined_r2_scores = combined_r2_scores.append(df["Score"], ignore_index=True)
                markers = df["Marker"]

        combined_r2_scores.columns = markers
        mean_scores = pd.DataFrame(columns=["Marker", "Score"],
                                   data={"Marker": combined_r2_scores.columns,
                                         "Score": combined_r2_scores.mean().values})

        # return mean scores
        return mean_scores, combined_r2_scores

    @staticmethod
    def load_r2_score(load_path: Path, file_name: str) -> Optional[pd.DataFrame]:
        """
        Loads the given r2 score file
        @param load_path: The path from where to load the data
        @param file_name: The file name to search for
        @return: Returns the r2 score file
        """

        for p in load_path.rglob("*"):
            if p.name == file_name:
                return pd.read_csv(p.absolute(), header=0)

        return None

    @staticmethod
    def load_file(load_path: Path, file_name: str = None) -> Optional[pd.DataFrame]:
        """
        Loads a file with the given name from the given path
        @param load_path: The path to search for the file
        @param file_name: File name with extensions
        @return: None or a loaded dataframe
        """

        if file_name is not None:
            for p in load_path.rglob("*"):
                if p.name == f"{file_name}":
                    df = pd.read_csv(p.absolute(), header=0)
                    return df

        else:
            return pd.read_csv(load_path, header=0)

        return None

    @staticmethod
    def load_layer_weights(load_path: Path, file_name: str) -> pd.DataFrame:
        """
        Loads all files matching the file name from the load path
        @param load_path: The path where files are being searched
        @param file_name: The file name pattern to match
        @return: A list of found dataframes
        """

        combined_weights = pd.DataFrame()

        for p in load_path.rglob("*"):
            if p.name != file_name:
                continue

            df = pd.read_csv(p.absolute(), header=0)

            combined_weights = combined_weights.append(df, ignore_index=True)

        return combined_weights

    @staticmethod
    def load_files_in_folder(folder: str, file_to_exclude: str, keep_spatial: bool = False) -> Tuple:
        """
        Loads all files in a folder, but excluding the file to exclude
        @param folder:
        @param file_to_exclude:
        @return: Returns a tuple. First is the dataset, second are the features, third is a list of all files used
        """
        files_used: list = []
        frames: list = []
        path_list = Path(folder).glob('**/*.csv')
        features: list = []

        found_excluded_file: bool = False

        for path in path_list:
            if "SARDANA" in path.stem:
                continue

            # Only raise an issue if the specific file cannot be found
            if Path(file_to_exclude).stem == path.stem:
                found_excluded_file = True
                continue

            cells, features = DataLoader.load_single_cell_data(file_name=str(path), keep_spatial=keep_spatial)
            frames.append(cells)
            files_used.append(path.stem)

        if not found_excluded_file:
            raise ValueError(f"File {file_to_exclude} could not be found! Please specify a valid path ")

        if len(frames) == 0:
            raise ValueError("No files found")

        data_set = pd.concat(frames)
        data_set.columns = features

        return data_set, features, files_used
