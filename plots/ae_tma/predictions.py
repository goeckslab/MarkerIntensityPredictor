import pandas as pd
import seaborn as sns
from pathlib import Path
import matplotlib.pyplot as plt
import argparse

SHARED_MARKERS = ['pRB', 'CD45', 'CK19', 'Ki67', 'aSMA', 'Ecad', 'PR', 'CK14', 'HER2', 'AR', 'CK17', 'p21', 'Vimentin',
                  'pERK', 'EGFR', 'ER']
save_path: Path = Path("plots", "figures", "supplements", "ae_tma")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--biopsy", help='Input file', required=True)
    args = parser.parse_args()
    biopsy: str = args.biopsy

    save_path = Path(save_path, biopsy)

    if not save_path.exists():
        save_path.mkdir(parents=True)

    # load ae tma scaled datasets scores from cleaned data tma folder
    ground_truth = pd.read_csv(Path("data", "cleaned_data", "tma", "scaled_biopsies", f"{biopsy}.tsv"), sep="\t")

    predictions = pd.read_csv(Path("data", "cleaned_data", "predictions", "ae_tma", "predictions.csv"))

    predictions = predictions[predictions["Biopsy"] == biopsy]

    # iterate thorugh all markers
    for marker in SHARED_MARKERS:
        gt = ground_truth[marker]
        pred = predictions[marker]

        sns.histplot(pred, color="orange", label="Predicted", kde=True)
        # scale y-axis of gt and train to match pred

        sns.histplot(gt, color="blue", label="Ground Truth", kde=True)
        # sns.histplot(train, color="green", label="TRAIN", kde=True)

        # change y axis label to cell count
        plt.ylabel("Cell Count")
        plt.xlabel(f"{marker} Expression")
        plt.legend()
        plt.savefig(Path(save_path, f"{marker}.png"), dpi=300, bbox_inches='tight')
        plt.close('all')
