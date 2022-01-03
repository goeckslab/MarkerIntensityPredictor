import pandas as pd
from pathlib import Path
import re


class Preprocessing:
    @staticmethod
    def get_data(input_dataframe: pd.DataFrame, keep_morph: bool = True):
        """
        Load Data
        We load data into RAM since data is small and will fit in memory.
        :return:
        """

        columns_to_remove = ["CellID", "ERK1_2_nucleiMasks"]
        cells = input_dataframe

        # Keeps only the 'interesting' columns with morphological features
        if keep_morph:
            print("Including morphological data")
            morph_data = pd.DataFrame(
                columns=["Area", "MajorAxisLength", "MinorAxisLength", "Eccentricity", "Solidity", "Extent",
                         "Orientation"])

            morph_data = cells.loc[:, morph_data.columns]

            cells = cells.filter(regex="nucleiMasks$", axis=1).filter(regex="^(?!(DAPI|AF))", axis=1)  # With morph data

            # Re add morph data
            for column in morph_data.columns:
                cells[f"{column}"] = morph_data[f"{column}"]

            if "Orientation" in cells.columns:
                cells['Orientation'] = cells["Orientation"].abs()

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

        if 'ERK1_1' in markers:
            assert 'ERK1_2' not in markers, 'ERK1_2 should not be in markers'

        # return cells, markers
        return cells.iloc[:, :], markers
