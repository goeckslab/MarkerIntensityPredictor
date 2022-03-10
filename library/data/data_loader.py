import pandas as pd
from pathlib import Path
import re
import os
from typing import Tuple


class DataLoader:
    @staticmethod
    def load_data(file_name: str, keep_morph: bool = True):
        """
        Loads the data, given the provided information
        @param file_name: The file name to load from the os
        @param keep_morph: Keep morphological features or sort them out? Default to true
        @return:
        """
        columns_to_remove = ["CellID", "ERK1_2_nucleiMasks"]

        path = Path(file_name)
        cells = pd.read_csv(path, header=0)

        # Keeps only the 'interesting' columns with morphological features
        if keep_morph:
            print("Including morphological data")
            morph_data = pd.DataFrame(
                columns=["Area", "MajorAxisLength", "MinorAxisLength", "Eccentricity", "Solidity", "Extent"])

            morph_data = cells.loc[:, morph_data.columns]

            cells = cells.filter(regex="nucleiMasks$", axis=1).filter(regex="^(?!(DAPI|AF))", axis=1)  # With morph data

            # Re add morph data
            for column in morph_data.columns:
                cells[f"{column}"] = morph_data[f"{column}"]


        # Keep only markers
        else:
            print("Excluding morphological data")
            cells = cells.filter(regex="nucleiMasks$", axis=1).filter(regex="^(?!(DAPI|AF))", axis=1)  # No morph data

        # Remove not required columns
        for column in columns_to_remove:
            if column in cells.columns:
                del cells[f"{column}"]

        markers = cells.columns
        markers = [re.sub("_nucleiMasks", "", x) for x in markers]

        assert 'ERK1_2' not in markers, 'ERK1_2 should not be in markers'
        assert 'CellId' not in markers, 'CellId should not be in markers'
        assert 'Orientation' not in markers, 'Orientation should not be in markers'

        # return cells, markers
        return cells.iloc[:, :], markers

    @staticmethod
    def load_r2_scores_for_model(load_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Loads all r2scores for the given model and combines them in a dataset
        @param load_path: The path from where to load the data
        @return: Returns a tuple of dataframes. First dataframe contains the mean scores, the second contains all combined scores.
        """
        combined_r2_scores = pd.DataFrame()
        markers = []

        for p in load_path.rglob("*"):
            if p.name == "r2_score.csv":
                df = pd.read_csv(p.absolute(), header=0)

                # Get markers
                markers = df["Marker"].to_list()

                # Transpose
                df = df.T
                # Drop markers row
                df.drop(index=df.index[0],
                        axis=0,
                        inplace=True)

                combined_r2_scores = combined_r2_scores.append(df, ignore_index=True)

        combined_r2_scores.columns = markers
        mean_scores = pd.DataFrame(columns=["Marker", "Score"],
                                   data={"Marker": combined_r2_scores.columns,
                                         "Score": combined_r2_scores.mean().values})

        # return mean scores
        return mean_scores, combined_r2_scores

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
