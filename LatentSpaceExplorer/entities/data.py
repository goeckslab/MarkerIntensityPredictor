import pandas as pd


class Data:
    generated_cells: pd.DataFrame
    latent_points: pd.DataFrame

    def __init__(self):
        self.generated_cells = pd.DataFrame()
        self.latent_points = pd.DataFrame()
