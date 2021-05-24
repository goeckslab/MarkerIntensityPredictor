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

        cells = pd.read_csv(Path(input_file), header=0)

        # Keeps only the 'interesting' columns.
        cells = cells.filter(regex="nucleiMasks$", axis=1).filter(regex="^(?!(DAPI|AF))", axis=1)
        markers = cells.columns
        markers = [re.sub("_nucleiMasks", "", x) for x in markers]

        return cells, markers

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
