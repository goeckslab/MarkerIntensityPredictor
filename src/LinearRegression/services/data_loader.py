import pandas as pd
import re
from pathlib import Path
import os


class DataLoader:

    @staticmethod
    def get_data(input_file: str):
        """
        Load Data
        We load data into RAM since data is small and will fit in memory.
        :return:
        """

        cells = pd.read_csv(Path(input_file), header=0)

        # Keeps only the 'interesting' columns.
        cells = cells.filter(regex="nucleiMasks$", axis=1).filter(regex="^(?!(DAPI|AF))", axis=1)
        markers = cells.columns
        markers = [re.sub("_nucleiMasks", "", x) for x in markers]

        return cells, markers

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
