import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from statannotations.Annotator import Annotator
import sys
import argparse

if __name__ == '__main__':
    df = pd.read_csv(Path("data", "cleaned_data", "null_model", "random_draw_all_predictions.csv"))

    df = df[df["Train"] == 1]
    # remove prediction with 0 and truth of 0
    df = df[(df["y_hat"] != 0) & (df["y_truth"] != 0)]


    # select only random rows from df, but include all columns, the null model and lgbm
    df = df.sample(n=1000, random_state=42)

    # calculate orrelation between y_hat and y_truth for null model == 0 and null model == 1
    lgbm_correlation = df[df["Null Model"] == 0][["y_hat", "y_truth"]].corr().iloc[0, 1]
    null_model_correlation = df[df["Null Model"] == 1][["y_hat", "y_truth"]].corr().iloc[0, 1]

    # create scatter plot
    sns.set_theme(style="whitegrid")
    sns.set_context("paper")
    sns.set(font_scale=1.5)
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.scatterplot(data=df, x="y_hat", y="y_truth", hue="Null Model")
    # change y and x axis labels
    plt.xlabel("Predicted expression")
    plt.ylabel("True expression")

    plt.title("Predictions vs. ground truth")
    # add correlation as text
    plt.text(0.05, 0.975, f"LGBM Corr: {round(lgbm_correlation, 2)}", transform=ax.transAxes, fontsize=14,
             verticalalignment='top')
    plt.text(0.3, 0.975, f"NM Corr: {round(null_model_correlation, 2)}", transform=ax.transAxes, fontsize=14,
             verticalalignment='top')
    # change legend names and place legend outside of plot
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=["LGBM", "Null Model"], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.tight_layout()

    save_path = Path("plots", "figures", "supplements", "null_model")
    if not save_path.exists():
        save_path.mkdir(parents=True)

    plt.savefig(Path(save_path, "predictions_vs_ground_truth.png"), dpi=300, bbox_inches='tight')
