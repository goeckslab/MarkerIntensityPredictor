from argparse import ArgumentParser
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SHARED_MARKER = ['pRB', 'CD45', 'CK19', 'Ki67', 'aSMA', 'Ecad', 'PR', 'CK14', 'HER2', 'AR', 'CK17', 'p21', 'Vimentin',
                 'pERK', 'EGFR', 'ER']

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--biopsy", "-b", type=str, required=True, help="The biopsy. Format should be 9_2_1")

    args = parser.parse_args()

    biopsy: str = args.biopsy
    patient: str = '_'.join(biopsy.split("_")[:2])
    preprocessed_intensity_marker = pd.read_csv(
        f"data/tumor_mesmer/combined/preprocessed/{patient}_excluded_dataset.tsv",
        sep="\t")
    intensity_marker = pd.read_csv(f"data/tumor_mesmer/combined/{patient}_excluded_dataset.csv", sep=",")[SHARED_MARKER]
    feature_importance = pd.read_csv(f"data/cleaned_data/feature_importance/{biopsy}_feature_attributions.csv", sep=",",
                                     index_col=0)
    # replace the nan values with 0
    feature_importance = feature_importance.replace(np.nan, 0)
    # remove biopsy column from df
    feature_importance = feature_importance.drop(columns=["Biopsy"])
    # create heatmap from df using seaborn
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, dpi=96, figsize=(10, 16))
    ax1.title.set_text(f"Feature importance for biopsy {biopsy}")
    sns.heatmap(feature_importance, annot=True, vmax=0.7, ax=ax1)
    ax2.title.set_text(f"Intensity marker for biopsy {biopsy}")
    sns.heatmap(intensity_marker.corr(), annot=True, ax=ax2)
    ax3.title.set_text(f"Preprocessed intensity marker for biopsy {biopsy}")
    sns.heatmap(preprocessed_intensity_marker.corr(), annot=True, ax=ax3)
    # plt.title(f"Feature importance for biopsy {biopsy}")
    plt.tight_layout()
    plt.savefig(f"plots/{biopsy}_feature_importance.png")
