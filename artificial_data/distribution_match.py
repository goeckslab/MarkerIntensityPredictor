import pandas as pd
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt

BIOPSIES = ["9_2_1", "9_2_2", "9_3_1", "9_3_2", "9_14_1", "9_14_2", "9_15_1", "9_15_2"]
SHARED_MARKERS = ['pRB', 'CD45', 'CK19', 'Ki67', 'aSMA', 'Ecad', 'PR', 'CK14', 'HER2', 'AR', 'CK17', 'p21', 'Vimentin',
                  'pERK', 'EGFR', 'ER']

if __name__ == '__main__':
    base_path = Path("artificial_data", "results", "distribution_match")

    predictions: pd.DataFrame = pd.read_csv(Path("artificial_data", "predictions.csv"))

    for correlation in predictions["Correlation"].unique():
        for biopsy in predictions["Biopsy"].unique():
            save_path = Path(base_path, str(correlation), biopsy)

            if not save_path.exists():
                save_path.mkdir(parents=True)

            # load ground truth
            ground_truth: pd.DataFrame = pd.read_csv(
                Path("data", "cleaned_data", "ground_truth", f"{biopsy}_preprocessed_dataset.tsv"), sep="\t")

            temp_predictions = predictions[predictions["Correlation"] == correlation]
            temp_predictions = temp_predictions[temp_predictions["Biopsy"] == biopsy]
            for marker in SHARED_MARKERS:
                gt_marker = ground_truth[marker]
                pred_marker = temp_predictions[marker]

                sns.histplot(pred_marker, color="orange", label="PRED", kde=True)
                # scale y-axis of gt and train to match pred

                sns.histplot(gt_marker, color="blue", label="GT", kde=True)
                # sns.histplot(train, color="green", label="TRAIN", kde=True)

                # change y axis label to cell count
                plt.ylabel("Cell Count")
                plt.xlabel(f"{marker} Expression")
                plt.legend()
                plt.savefig(Path(save_path, f"{marker}.png"))
                plt.close('all')
