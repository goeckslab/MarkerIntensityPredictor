import pandas as pd
from typing import List


class PhenotypeMapper:

    @staticmethod
    def map_nn_to_phenotype(nearest_neighbors: pd.DataFrame, phenotypes: pd.DataFrame,
                            neighbor_count: int) -> pd.DataFrame:
        nearest_neighbors = nearest_neighbors.iloc[1:, :].copy()
        nearest_neighbors_lists = nearest_neighbors.apply(PhenotypeMapper.convert_index_to_phenotype,
                                                          phenotypes=phenotypes, axis=1)

        for i in range(neighbor_count):
            nearest_neighbors.iloc[i] = nearest_neighbors_lists.iloc[i]

        return nearest_neighbors

    @staticmethod
    def convert_index_to_phenotype(x, phenotypes: pd.DataFrame):
        return pd.Series([phenotypes.iloc[index].values[0] for index in x])
