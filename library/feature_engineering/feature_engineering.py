import pandas as pd
from tqdm import tqdm
from sklearn.neighbors import BallTree
from pathlib import Path
from typing import Dict, List
from library import DataLoader, Preprocessing
from sklearn.preprocessing import OneHotEncoder


class FeatureEngineer:

    def __init__(self, radius: int, folder_name: str = None, file_to_exclude: str = None, file_path: str = None):
        self._folder_name: str = folder_name
        self._file_to_exclude: str = file_to_exclude
        self._radius: int = radius

        if self._folder_name is not None:
            self._path_list: List = list(Path(self._folder_name).glob('**/*.csv'))
        self._file_path: str = file_path
        self._feature_engineered_data: Dict = {}
        self._phenotypes: Dict = {}
        self._prepared_data: Dict = {}
        self._marker_columns: List = []

        if file_to_exclude is None and file_path is None:
            raise ValueError(
                "Please provide either a file to process or a file to exclude combining with a folder to process or a dataset")

    @property
    def feature_engineered_data(self):
        return self._feature_engineered_data

    @property
    def radius(self):
        return self._radius

    @radius.setter
    def radius(self, value):
        self._radius = value

    @property
    def phenotypes(self):
        return self._phenotypes

    @property
    def file_path(self):
        return self.file_path

    @property
    def folder_name(self):
        return self._folder_name

    @property
    def excluded_file(self):
        return self._file_to_exclude

    @property
    def marker_columns(self):
        return self._marker_columns

    def create_features(self, add_enhanced_features: bool = False):
        # Reset results
        self._feature_engineered_data = {}
        self._phenotypes = {}
        self._marker_columns = []

        if self._file_path is not None:
            self.__prepare_data(self._file_path)
            self.__engineer_new_features(Path(self._file_path))
            return

        for path in self._path_list:
            if "SARDANA" in path.stem:
                continue

            # Only raise an issue if the specific file cannot be found
            if Path(self._file_to_exclude).stem == path.stem:
                continue

            self.__prepare_data(str(path))
            self.__engineer_new_features(path, add_enhanced_features=add_enhanced_features)

    def __prepare_data(self, file_name: str):
        path = Path(file_name)
        data, self._marker_columns = DataLoader.load_single_cell_data(str(file_name), keep_spatial=True,
                                                                      return_df=True,
                                                                      return_marker_columns=True)
        coords = data[["X_centroid", "Y_centroid"]].copy()
        # data = Preprocessing.normalize(data=data, columns=data.columns)
        data["X_centroid"] = coords["X_centroid"].values
        data["Y_centroid"] = coords["Y_centroid"].values

        data["Phenotype"] = pd.read_csv(
            f"{Path(__file__).parent.parent.parent}/phenotypes/{path.stem}_phenotypes.csv")["phenotype"].values

        self._prepared_data[path.stem] = data
        self._phenotypes[path.stem] = data["Phenotype"]

    def __engineer_new_features(self, file_path: Path, add_enhanced_features: bool = False):
        """
        New features are created here
        @param file_path:
        @return:
        """
        file_name: str = file_path.stem
        data = self._prepared_data.get(file_name).copy()

        print("Creating new features")
        tree = BallTree(data[["X_centroid", "Y_centroid"]], leaf_size=2)
        indexes, distances = tree.query_radius(data[["X_centroid", "Y_centroid"]], r=self._radius,
                                               return_distance=True, sort_results=True)

        indexes = [index for index in indexes if len(index) > 1]

        # Extend dataframe
        mean_marker_columns = [f"Mean Neighbor Intensity {marker}" for marker in self._marker_columns]
        mean_marker_columns.extend(data.columns)
        mean_marker_columns.append("# of Immune Cells")
        mean_marker_columns.append("# of Neoplastic Epithelial Cells")
        mean_marker_columns.append("# of Stroma Cells")

        data = data.reindex(columns=mean_marker_columns, fill_value=0)

        if add_enhanced_features:
            max_marker_columns = [f"Max Neighbor Intensity {marker}" for marker in self._marker_columns]
            max_marker_columns.extend(data.columns)
            data = data.reindex(columns=max_marker_columns, fill_value=0)

            min_marker_columns = [f"Min Neighbor Intensity {marker}" for marker in self._marker_columns]
            min_marker_columns.extend(data.columns)
            data = data.reindex(columns=min_marker_columns, fill_value=0)

            median_marker_columns = [f"Median Neighbor Intensity {marker}" for marker in self._marker_columns]
            median_marker_columns.extend(data.columns)
            data = data.reindex(columns=median_marker_columns, fill_value=0)

        for i, index in tqdm(enumerate(indexes)):

            cells = data.iloc[index]
            base_cell = cells[:1]
            neighbor_cells = cells[1:]

            phenotype_counts = neighbor_cells["Phenotype"].value_counts()

            data.at[base_cell.index[0], "Cell Neighborhood"] = self.__most_frequent(
                list(neighbor_cells["Phenotype"].values))

            data.at[base_cell.index[0], "# of Immune Cells"] = phenotype_counts[
                'Immune'] if "Immune" in phenotype_counts else 0

            neo_count = phenotype_counts[
                'Neoplastic Epithelial'] if "Neoplastic Epithelial" in phenotype_counts else 0
            luminal_count = phenotype_counts['Luminal'] if "Luminal" in phenotype_counts else 0
            basal_count = phenotype_counts['Basal'] if "Basal" in phenotype_counts else 0
            data.at[
                base_cell.index[0], "# of Neoplastic Epithelial Cells"] = neo_count + luminal_count + basal_count

            data.at[base_cell.index[0], "# of Stroma Cells"] = phenotype_counts[
                'Stroma (aSMA+)'] if "Stroma (aSMA+)" in phenotype_counts else 0

            # Update value for new column
            for marker in self._marker_columns:
                data.at[base_cell.index[0], f"Mean Neighbor Intensity {marker}"] = neighbor_cells[
                    marker].mean()

                if add_enhanced_features:
                    data.at[base_cell.index[0], f"Max Neighbor Intensity {marker}"] = neighbor_cells[
                        marker].max()

                    data.at[base_cell.index[0], f"Min Neighbor Intensity {marker}"] = neighbor_cells[
                        marker].min()

                    data.at[base_cell.index[0], f"Median Neighbor Intensity {marker}"] = neighbor_cells[
                        marker].median()

        data["Cell Neighborhood"].fillna("Unknown", inplace=True)

        # One Hot Encoding of nominal cell neighborhood
        enc = OneHotEncoder(handle_unknown='ignore')
        one_hot_encoded = enc.fit_transform(data[['Cell Neighborhood']]).toarray()
        enc_df = pd.DataFrame(data=one_hot_encoded,
                              columns=[f"{feature_name.split('_')[-1]} Cell Neighborhood" for feature_name in
                                       enc.get_feature_names_out()])
        data = data.join(enc_df)

        # One Hot Encoding of nominal phenotype
        one_hot_encoded = enc.fit_transform(data[['Phenotype']]).toarray()
        enc_df = pd.DataFrame(data=one_hot_encoded,
                              columns=[f"{feature_name.split('_')[-1]} Phenotype" for feature_name in
                                       enc.get_feature_names_out()])
        data = data.join(enc_df)

        # Drop not required data such as phenotype etc
        data.drop(columns=["Phenotype", "Cell Neighborhood"], inplace=True)

        self._feature_engineered_data[file_name] = data

    def __most_frequent(self, cell_neighbor_hood: List):
        return max(set(cell_neighbor_hood), key=cell_neighbor_hood.count)
