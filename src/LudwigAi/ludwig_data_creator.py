from pathlib import Path
from shared.services.data_loader import DataLoader
import re


class LudwigAi:

    def __init__(self, file: Path):
        inputs, markers = DataLoader.get_data(str(file))
        print(markers)
        inputs.columns = [re.sub("_nucleiMasks", "", x) for x in inputs.columns]

        inputs.to_csv(Path(f"results/ludwig_data.txt"), index= False, sep=',')
