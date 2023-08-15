import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from statannotations.Annotator import Annotator
import sys


def create_bar_chart(data: pd.DataFrame):
    print(data)
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

    save_path = Path("plots", "figures", "null_model")
    if not save_path.exists():
        save_path.mkdir(parents=True)

    plt.savefig(Path(save_path, "random_draw_vs_lgbm_model_performance.png"), dpi=300)


if __name__ == '__main__':
    df = pd.read_csv(Path("data", "cleaned_data", "null_model", "random_draw_all_predictions.csv"))

    # select only train data
    df = df[df["Train"] == 1]
    df = df[df["Null Model"] == 1]

    print(df)


    # create scatter plot
    sns.set_theme(style="whitegrid")
    sns.set_context("paper")
    sns.set(font_scale=1.5)
    sns.scatterplot(data=df, x="y_hat", y="y_truth", hue="Null Model")
    plt.show()

