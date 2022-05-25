import pandas as pd
from typing import List


class PhenotypeMapper:

    @staticmethod
    def map_nn_to_phenotype(nearest_neighbors: pd.DataFrame, phenotypes: pd.DataFrame) -> pd.DataFrame:
        nearest_neighbors = nearest_neighbors.iloc[1:, :].copy()
        nearest_neighbors_lists = nearest_neighbors.apply(PhenotypeMapper.test_lambda, phenotypes=phenotypes, axis=1)

        nearest_neighbors.iloc[0] = nearest_neighbors_lists.iloc[0]
        nearest_neighbors.iloc[1] = nearest_neighbors_lists.iloc[1]

        return nearest_neighbors

    @staticmethod
    def test_lambda(x, phenotypes: pd.DataFrame):
        # print("Show x")
        # print(x)
        # print(phenotypes)
        # print([phenotypes.iloc[index].values[0] for index in x])
        # input()

        return pd.Series([phenotypes.iloc[index].values[0] for index in x])
