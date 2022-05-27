import scimap as sm
import pandas as pd
from library import DataLoader, FolderManagement, Preprocessing
import anndata as ad
import os
import argparse
from pathlib import Path
from typing import List

results_folder = Path("phenotyping")


def get_args():
    """
    Load all provided cli args
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", "-f", action="store", required=True, type=str, help="The file to phenotype")

    return parser.parse_args()


args = get_args()

if not results_folder.exists():
    FolderManagement.create_directory(path=results_folder)

cells: pd.DataFrame = DataLoader.load_single_cell_data(file_name=args.file, return_df=True)

# phenotype_workflow: pd.DataFrame = pd.read_csv("phenotype_workflow.csv")
# del phenotype_workflow["Unnamed: 0"]

columns = ["Unnamed: 0", "Unnamed: 1"]
columns.extend(cells.columns)

# phenotype_workflow: pd.DataFrame = pd.DataFrame(columns=columns)

conditions: List = [
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
        "Unnamed: 1": "Basal",
        "CK14": "anypos",
        "CK17": "anypos",
    },

    {
        "Unnamed: 0": "Neoplastic Epithelial",
        "Unnamed: 1": "Luminal",
        "CK19": "anypos",
        "CK7": "anypos",
    },

    {
        "Unnamed: 0": "all",
        "Unnamed: 1": "Immune",
        "CD45": "anypos"
    }]

phenotype_workflow = pd.DataFrame(columns=columns).from_records(conditions)

adata = ad.AnnData(cells)

file_name: str = Path(args.file).name

# phenotype = pd.read_csv('path/to/csv/file/')

adata = sm.tl.phenotype_cells(adata, phenotype=phenotype_workflow, label="phenotype")

adata.obs.to_csv(f"{os.path.join(results_folder, file_name)}_phenotypes.csv", index=False)
