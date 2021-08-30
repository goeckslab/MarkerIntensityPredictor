import pandas as pd
from pathlib import Path
import re
import os


class DataLoader:
    @staticmethod
    def get_data(input_file: str, keep_morph: bool):
        """
        Load Data
        We load data into RAM since data is small and will fit in memory.
        :return:
        """

        columns_to_remove = ["CellID", "ERK1_2_nucleiMasks"]

        path = Path(f"{os.path.split(os.environ['VIRTUAL_ENV'])[0]}/{input_file}")
        cells = pd.read_csv(path, header=0)

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

        # return cells, markers
        return cells.iloc[:, :], markers

    @staticmethod
    def load_folder_data(path: str, keep_morph: bool):
        merged_data = pd.DataFrame()
        markers = list
        for subdir, dirs, files in os.walk(path):
            for file in files:
                if not file.endswith(".csv"):
                    continue

                cells, markers = DataLoader.get_data(os.path.join(subdir, file), keep_morph)
                merged_data = merged_data.append(cells)

        return merged_data, markers

    @staticmethod
    def merge_files(exclude_file: str):
        """
        Combines all remaining files in the folder to create a big train set
        :param exclude_file:
        :return:
        """

        path: Path = Path("./data")
        merged_data = pd.DataFrame()
        markers = list
        for subdir, dirs, files in os.walk(path):
            for file in files:
                if not file.endswith(".csv"):
                    continue

                if file in str(exclude_file):
                    continue

                cells, markers = DataLoader.get_data(os.path.join(subdir, file), True)
                merged_data = merged_data.append(cells)

        return merged_data, markers
