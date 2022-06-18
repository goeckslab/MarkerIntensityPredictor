import pandas as pd
from tqdm import tqdm
from sklearn.neighbors import BallTree
from pathlib import Path
from typing import Dict, List
from library import DataLoader, Preprocessing
from sklearn.preprocessing import LabelEncoder


class FeatureEngineer:

    def __init__(self, radius: int, folder: str = None, file_to_exclude: str = None, file: str = None):
        self._folder: str = folder
        self._file_to_exclude: str = file_to_exclude
        self._radius: int = radius

        if self._folder is not None:
            self._path_list: List = list(Path(self._folder).glob('**/*.csv'))
        self._file: str = file
        self._results: Dict = {}
        self._phenotypes: Dict = {}

        if file_to_exclude is None and file is None:
            raise ValueError(
                "Please provide either a file to process or a file to exclude combining with a folder to process")

    @property
    def results(self):
        return self._results

    @property
    def radius(self):
        return self._radius

    @radius.setter
    def radius(self, value):
        self._radius = value

    @property
    def phenotypes(self):
        return self._phenotypes

    def start_processing(self):
        # Reset results
        self._results = {}
        self._phenotypes = {}

        if self._file is not None:
            self.__prepare_data(self._file)
            return

        for path in self._path_list:
            if "SARDANA" in path.stem:
                continue

            # Only raise an issue if the specific file cannot be found
            if Path(self._file_to_exclude).stem == path.stem:
                continue

            self.__prepare_data(str(path))

    def __prepare_data(self, file_name: str):
        path = Path(file_name)
        data, marker_columns = DataLoader.load_single_cell_data(str(file_name), keep_spatial=True, return_df=True,
                                                                return_marker_columns=True)

        coords = data[["X_centroid", "Y_centroid"]].copy()
        data = Preprocessing.normalize(data=data, create_dataframe=True, columns=data.columns)
        data["X_centroid"] = coords["X_centroid"].values
        data["Y_centroid"] = coords["Y_centroid"].values

        data["Phenotype"] = pd.read_csv(
            f"{Path(__file__).parent.parent.parent}/phenotypes/{path.stem}_phenotypes.csv")["phenotype"].values
        # Create new features
        print("Creating new features...")
        data = self.__create_new_features(data=data, marker_columns=marker_columns)
        self._results[path.stem] = data
        self._phenotypes[path.stem] = data["Phenotype"]

    def __create_new_features(self, data: pd.DataFrame, marker_columns: List) -> pd.DataFrame:
        tree = BallTree(data[["X_centroid", "Y_centroid"]], leaf_size=2)
        indexes, distances = tree.query_radius(data[["X_centroid", "Y_centroid"]], r=self._radius,
                                               return_distance=True, sort_results=True)

        indexes = [index for index in indexes if len(index) > 1]

        # Extend dataframe
        mean_marker_columns = [f"Mean Neighbor Intensity {marker}" for marker in marker_columns]
        mean_marker_columns.extend(data.columns)
        mean_marker_columns.append("# of Immune Cells")
        mean_marker_columns.append("# of Neoplastic Epithelial Cells")
        mean_marker_columns.append("# of Stroma Cells")
        data = data.reindex(columns=mean_marker_columns, fill_value=0)

        for i, index in tqdm(enumerate(indexes)):

            cells = data.iloc[index]
            base_cell = cells[:1]
            neighbor_cells = cells[1:]

            phenotype_counts = neighbor_cells["Phenotype"].value_counts()

            data.at[base_cell.index[0], "Cell Neighborhood"] = self.__most_frequent(
                list(neighbor_cells["Phenotype"].values))

            data.at[base_cell.index[0], "# of Immune Cells"] = phenotype_counts[
                'Immune'] if "Immune" in phenotype_counts else 0

            neo_count = phenotype_counts['Neoplastic Epithelial'] if "Neoplastic Epithelial" in phenotype_counts else 0
            luminal_count = phenotype_counts['Luminal'] if "Luminal" in phenotype_counts else 0
            basal_count = phenotype_counts['Basal'] if "Basal" in phenotype_counts else 0
            data.at[base_cell.index[0], "# of Neoplastic Epithelial Cells"] = neo_count + luminal_count + basal_count

            data.at[base_cell.index[0], "# of Stroma Cells"] = phenotype_counts[
                'Stroma (aSMA+)'] if "Stroma (aSMA+)" in phenotype_counts else 0

            # Update value for new column
            for marker in marker_columns:
                data.at[base_cell.index[0], f"Mean Neighbor Intensity {marker}"] = neighbor_cells[
                    marker].mean()

        data["Cell Neighborhood"].fillna("Unknown", inplace=True)

        le = LabelEncoder()
        data["Cell Neighborhood Encoded"] = le.fit_transform(
            data["Cell Neighborhood"])

        return data

    def __most_frequent(self, cell_neighbor_hood: List):
        return max(set(cell_neighbor_hood), key=cell_neighbor_hood.count)
