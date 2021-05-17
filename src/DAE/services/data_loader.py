import pandas as pd
from pathlib import Path
import re


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

        #return cells, markers
        return cells.iloc[:, :], markers
