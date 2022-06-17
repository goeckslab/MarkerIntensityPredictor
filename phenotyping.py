import scimap as sm
import pandas as pd
from library import DataLoader, FolderManagement
import anndata as ad
import os
import argparse
from pathlib import Path
from typing import List

results_folder = Path("phenotypes")


def get_args():
    """
    Load all provided cli args
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", "-f", action="store", required=True, type=str, help="The file to phenotype")
    # parser.add_argument("--imageId", "-iid", action="store", required=True, type=str,
    #                    help="The image id being processed")

    return parser.parse_args()


args = get_args()

if not results_folder.exists():
    FolderManagement.create_directory(path=results_folder)

cells: pd.DataFrame = DataLoader.load_single_cell_data(file_name=args.file, return_df=True)

columns = ["Unnamed: 0", "Unnamed: 1"]
columns.extend(cells.columns)

conditions: List = [
    {
        "Unnamed: 0": "all",
        "Unnamed: 1": "Immune",
        "CD45": "pos"
    },
    {
        "Unnamed: 0": "all",
        "Unnamed: 1": "Stroma",
        "aSMA": "pos",
    },
    {
        "Unnamed: 0": "all",
        "Unnamed: 1": "Neoplastic Epithelial",
        "CK19": "anypos",
        "CK14": "anypos",
        "CK17": "anypos",
        "CK7": "anypos"
    },
    {
        "Unnamed: 0": "Neoplastic Epithelial",
        "Unnamed: 1": "Luminal",
        "CK19": "anypos",
        "CK7": "anypos",
    },
    {
        "Unnamed: 0": "Neoplastic Epithelial",
        "Unnamed: 1": "Basal",
        "CK14": "anypos",
        "CK17": "anypos",
        "Ecad": "anypos"
    }
]

phenotype_workflow = pd.DataFrame(columns=columns).from_records(conditions)

adata = ad.AnnData(cells)
file_name: str = Path(args.file).stem
adata.obs['imageid'] = file_name.split('_')[-1]
adata = sm.pp.rescale(adata)

phenotype = pd.read_csv("/Users/kirchgae/Downloads/tumor_phenotypes.csv", sep=',')
adata = sm.tl.phenotype_cells(adata, phenotype=phenotype, label="phenotype")

print(adata.obs['phenotype'].value_counts())
adata.obs.to_csv(f"{os.path.join(results_folder, file_name)}_phenotypes.csv", index=False)
