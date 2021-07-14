import pandas as pd
from pathlib import Path
import re
import os


class DataLoader:
    @staticmethod
    def get_data(input_file: str):
        """
        Load Data
        We load data into RAM since data is small and will fit in memory.
        :return:
        """

        path = Path(f"{os.path.split(os.environ['VIRTUAL_ENV'])[0]}/{input_file}")
        cells = pd.read_csv(path, header=0)

        # Keeps only the 'interesting' columns.
        cells = cells.filter(regex="nucleiMasks$", axis=1).filter(regex="^(?!(DAPI|AF))", axis=1)
        # cells.columns = [re.sub("_nucleiMasks", "", x) for x in cells.columns]
        # cells = cells.filter(['HER2', 'CK7', 'CK17', 'cPARP', 'AR', 'CK14', 'p21', 'CK19', 'EGFR', 'Ki67'])

        if "ERK1_2_nucleiMasks" in cells.columns:
            del cells["ERK1_2_nucleiMasks"]

        markers = cells.columns
        markers = [re.sub("_nucleiMasks", "", x) for x in markers]

        # return cells, markers
        return cells.iloc[:, :], markers

    @staticmethod
    def load_folder_data(path: str):
        merged_data = pd.DataFrame()
        markers = list
        for subdir, dirs, files in os.walk(path):
            for file in files:
                if not file.endswith(".csv"):
                    continue

                cells, markers = DataLoader.get_data(os.path.join(subdir, file))
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

                cells, markers = DataLoader.get_data(os.path.join(subdir, file))
                merged_data = merged_data.append(cells)

        return merged_data, markers
