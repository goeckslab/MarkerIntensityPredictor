import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from statannotations.Annotator import Annotator
import sys


def create_bar_chart(data: pd.DataFrame):
    hue = "Model"
    hue_order = ["LGBM", "Null Model"]
    pairs = [
        (("AR", "LGBM"), ("AR", "Null Model")),
        (("CK14", "LGBM"), ("CK14", "Null Model")),
        (("CK17", "LGBM"), ("CK17", "Null Model")),
        (("CK19", "LGBM"), ("CK19", "Null Model")),
        (("CD45", "LGBM"), ("CD45", "Null Model")),
        (("Ecad", "LGBM"), ("Ecad", "Null Model")),
        (("EGFR", "LGBM"), ("EGFR", "Null Model")),
        (("ER", "LGBM"), ("ER", "Null Model")),
        (("HER2", "LGBM"), ("HER2", "Null Model")),
        (("Ki67", "LGBM"), ("Ki67", "Null Model")),
        (("PR", "LGBM"), ("PR", "Null Model")),
        (("Vimentin", "LGBM"), ("Vimentin", "Null Model")),
        (("aSMA", "LGBM"), ("aSMA", "Null Model")),
        (("p21", "LGBM"), ("p21", "Null Model")),
        (("pERK", "LGBM"), ("pERK", "Null Model")),
        (("pRB", "LGBM"), ("pRB", "Null Model")),

    ]

    fig, ax = plt.subplots(figsize=(15, 10))
    sns.boxenplot(x="Marker", y="MAE", data=mean_df, ax=ax, hue=hue)

    order = ["AR", "CK14", "CK17", "CK19", "CD45", "Ecad", "EGFR", "ER", "HER2", "Ki67", "PR", "Vimentin", "aSMA",
             "p21", "pERK", "pRB"]
    annotator = Annotator(ax, pairs, data=data, x="Marker", y="MAE", order=order, hue=hue, hue_order=hue_order,
                          verbose=1)
    annotator.configure(test='Mann-Whitney', text_format='star', loc='outside')
    annotator.apply_and_annotate()

    plt.title("Null model vs LGBM model performance", x=0.5, y=1.10, fontsize=20)
    # plot bar plot for mean_df
    sns.set_theme(style="whitegrid")
    sns.set_context("paper")
    sns.set(font_scale=1.5)

    save_path = Path("plots", "figures", "supplements", "null_model")
    if not save_path.exists():
        save_path.mkdir(parents=True)

    plt.savefig(Path(save_path, "performance.png"), dpi=300, bbox_inches='tight')


if __name__ == '__main__':
    df = pd.read_csv(Path("data", "cleaned_data", "null_model", "random_draw_all_performance.csv"))

    # select only train data
    df = df[df["Train"] == 1]

    # create new df and pair mae values of null model and biopsy model in two columns
    new_df = df[["Biopsy", "Marker", "MAE", "Null Model"]]

    # create new df that puts null model and biopsy model mae values in two columns
    new_df = new_df.pivot_table(index=["Biopsy", "Marker"], columns="Null Model", values="MAE")
    # create new columns using the biopsy
    new_df["Biopsy"] = new_df.index.get_level_values(0)
    # rename to 0 biopsy model and 1 to null model
    new_df.rename(columns={0: "Biopsy Model MAE", 1: "Null Model MAE"}, inplace=True)
    # convert index marker to column marker
    new_df.reset_index(level="Marker", inplace=True)
    # drop index
    new_df.reset_index(drop=True, inplace=True)

    # calculate mean performance for markers for both biopsy model and mull model
    # mean_df = new_df.groupby("Marker").mean()

    # reset indx
    # mean_df.reset_index(inplace=True)

    # Combine columns Biopsy Model MAE and NUll Model MAE to be one column
    mean_df = new_df.melt(id_vars=["Marker"], value_vars=["Biopsy Model MAE", "Null Model MAE"])
    # rename value column to MAE
    mean_df.rename(columns={"value": "MAE"}, inplace=True)
    # rename Null Model to Model
    mean_df.rename(columns={"Null Model": "Model"}, inplace=True)
    # rename Biopsy Model MAE to LGBM
    mean_df["Model"].replace({"Biopsy Model MAE": "LGBM"}, inplace=True)
    # rename Null Model MAE to Null Model
    mean_df["Model"].replace({"Null Model MAE": "Null Model"}, inplace=True)

    create_bar_chart(data=mean_df)
    sys.exit(0)
