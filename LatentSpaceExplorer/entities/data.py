import pandas as pd


class Data:
    generated_cells: pd.DataFrame
    latent_points: list

    def __init__(self):
        self.generated_cells = pd.DataFrame()
        self.latent_points = []
