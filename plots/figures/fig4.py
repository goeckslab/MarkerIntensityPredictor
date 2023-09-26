import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path
import os, sys, logging
from typing import List
from statannotations.Annotator import Annotator

logging_path = Path("plots", "figures", "fig4.log")
logging.root.handlers = []
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler(logging_path),
                        logging.StreamHandler()
                    ])


def create_boxen_plot(data: pd.DataFrame, metric: str, ylim: List, microns: List):
    color_palette = {"0 µm": "grey", "15 µm": "magenta", "30 µm": "purple", "60 µm": "green", "90 µm": "yellow",
                     "120 µm": "blue"}

    hue = "FE"
    hue_order = microns
    ax = sns.boxenplot(data=data, x="Marker", y=metric, hue=hue, palette=color_palette)

    plt.ylabel("")
    plt.xlabel("")

    plt.box(False)
    # remove legend from fig
    plt.legend(bbox_to_anchor=[0.5, 0.9], loc='center', fontsize=7, ncol=4)

    # reduce font size of x and y ticks
    ax.tick_params(axis='both', which='major', labelsize=8)
    plt.ylim(0, 0.4)

    pairs = []
    for micron in microns:
        if micron == "0 µm":
            continue
        # Create pairs of (micron, 0 µm) for each marker
        for marker in data["Marker"].unique():
            pairs.append(((marker, micron), (marker, "0 µm")))

    try:
        order = ['pRB', 'CD45', 'CK19', 'Ki67', 'aSMA', 'Ecad', 'PR', 'CK14', 'HER2', 'AR', 'CK17', 'p21',
                 'Vimentin',
                 'pERK', 'EGFR', 'ER']
        annotator = Annotator(ax, pairs, data=data, x="Marker", y=metric, order=order, hue=hue, hue_order=hue_order,
                              hide_non_significant=True)
        annotator.configure(test='Mann-Whitney', text_format='star', loc='outside')
        annotator.apply_and_annotate()

    except:
        logging.error(pairs)
        logging.error(data["FE"].unique())
        raise

    # plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    return ax


if __name__ == '__main__':
    if logging_path.exists():
        os.remove(logging_path)
    save_path = Path("images", "fig4")

    lgbm_scores = pd.read_csv(Path("data", "cleaned_data", "scores", "lgbm", "scores.csv"))

    # select only the scores for the 0 µm, 23 µm, 92 µm, 184 µm
    lgbm_scores = lgbm_scores[lgbm_scores["FE"].isin([0, 23, 92, 184])]
    # select exp scores
    lgbm_scores = lgbm_scores[lgbm_scores["Mode"] == "EXP"]
    # only select non hp scores
    lgbm_scores = lgbm_scores[lgbm_scores["HP"] == 0]

    # Add µm to the FE column
    lgbm_scores["FE"] = lgbm_scores["FE"].astype(str) + " µm"
    lgbm_scores["FE"] = pd.Categorical(lgbm_scores['FE'], ["0 µm", "23 µm", "92 µm", "184 µm"])

    # update 23 to 15, 92 to 60 and 184 to 120
    lgbm_scores["FE"] = lgbm_scores["FE"].cat.rename_categories(["0 µm", "15 µm", "60 µm", "120 µm"])

    # sort by marker and FE
    lgbm_scores.sort_values(by=["Marker", "FE"], inplace=True)

    # load image from images fig3 folder
    spatial_information_image = plt.imread(Path("images", "fig4", "panel_a.png"))

    dpi = 300
    cm = 1 / 2.54  # centimeters in inches
    # Create new figure
    fig = plt.figure(figsize=(18.5 * cm, 12 * cm), dpi=dpi)
    gspec = fig.add_gridspec(2, 3)

    ax1 = fig.add_subplot(gspec[0, :2])
    # remove box from ax1
    plt.box(False)
    # remove ticks from ax1
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.text(-0.05, 1.1, "a", transform=ax1.transAxes,
             fontsize=7, fontweight='bold', va='top', ha='right')
    # show spatial information image
    ax1.imshow(spatial_information_image, aspect='auto')

    ax2 = fig.add_subplot(gspec[1, :])
    ax2.set_title('LGBM 0 vs. 15 µm, 60 µm and 120 µm', rotation='vertical', x=-0.07, y=-0.2, fontsize=7)
    ax2.text(-0.05, 1.2, "b", transform=ax2.transAxes,
             fontsize=7, fontweight='bold', va='top', ha='right')
    # remove box from ax3
    plt.box(False)
    ax2 = create_boxen_plot(data=lgbm_scores, metric="MAE", ylim=[0, 0.5],
                            microns=["0 µm", "15 µm", "60 µm", "120 µm"])

    plt.tight_layout()
    plt.savefig(Path(save_path, "fig4.png"), dpi=300, bbox_inches='tight')
    plt.savefig(Path(save_path, "fig4.eps"), dpi=300, bbox_inches='tight', format='eps')
