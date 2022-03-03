import pandas as pd
from pathlib import Path
import re
import os
from typing import Tuple


class DataLoader:
    @staticmethod
    def load_data(source_df: pd.DataFrame = None, file_name: str = None, keep_morph: bool = True):
        """
        Loads the data, given the provided information
        @param source_df: An initial source dataframe already stored in memory
        @param file_name: The file name to load from the os
        @param keep_morph: Keep morphological features or sort them out? Default to true
        @return:
        """
        columns_to_remove = ["CellID", "ERK1_2_nucleiMasks"]

        if file_name is None and source_df is None:
            raise "Please provide either an input file or an source_df"

        if file_name is not None:
            path = Path(file_name)
            cells = pd.read_csv(path, header=0)
        else:
            cells = source_df

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

        assert 'ERK1_2' not in markers, 'ERK1_2 should not be in markers'
        assert 'CellId' not in markers, 'CellId should not be in markers'

        # return cells, markers
        return cells.iloc[:, :], markers
