import argparse, os, sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Tuple
from statannotations.Annotator import Annotator


def create_boxen_plot(results: pd.DataFrame, metric: str, save_path: Path, model: str):
    fig = plt.figure(figsize=(10, 10), dpi=200)
    ax = sns.boxenplot(x="Quartile", y=metric, hue="Mode", data=results, palette={"IP": "lightblue", "EXP": "orange"})
    ax.set_xlabel("Quartile")
    ax.set_ylabel(metric.upper())
    ax.tick_params(axis='both', which='major', labelsize=16)
    plt.box(False)
    # remove x and y label
    ax.set_xlabel("")
    ax.set_ylabel("")

    # ax.set_title(f"Elastic Net \n{metric.upper()} per quartile\nAll Biopsies", fontsize=20, y=1.3)

    hue = "Mode"
    hue_order = ["IP", "EXP"]
    pairs = [
        (("Q1", "IP"), ("Q2", "IP")),
        (("Q2", "IP"), ("Q3", "IP")),
        (("Q3", "IP"), ("Q4", "IP")),
        (("Q1", "EXP"), ("Q2", "EXP")),
        (("Q2", "EXP"), ("Q3", "EXP")),
        (("Q3", "EXP"), ("Q4", "EXP")),
    ]
    order = ["Q1", "Q2", "Q3", "Q4"]
    annotator = Annotator(ax, pairs, data=results, x="Quartile", y=metric, order=order, hue=hue, hue_order=hue_order,
                          verbose=1)
    annotator.configure(test='Mann-Whitney', text_format='star', loc='outside')
    annotator.apply_and_annotate()
    plt.legend().set_visible(False)
    plt.tight_layout()
    plt.savefig(str(Path(save_path, f"{model}.png")))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", action="store", choices=["EN", "LGBM", "AE", "AE M", "VAE ALL", "GNN"])
    args = parser.parse_args()

    save_path = Path("plots", "figures", "supplements", "quartile_performance")

    if not save_path.exists():
        save_path.mkdir(parents=True)

    model: str = args.model

    if model == "EN":
        pass
    elif model == "LGBM":
        quartile_performance = pd.read_csv(
            Path("data", "cleaned_data", "quartile_performance", "lgbm", "quartile_performance.csv"))
        create_boxen_plot(quartile_performance, "MAE", save_path=save_path, model="lgbm")
    elif model == "AE":
        quartile_performance = pd.read_csv(
            Path("data", "cleaned_data", "quartile_performance", "ae", "quartile_performance.csv"))
        create_boxen_plot(quartile_performance, "MAE", save_path=save_path, model="ae")
    elif model == "AE M":
        quartile_performance = pd.read_csv(
            Path("data", "cleaned_data", "quartile_performance", "ae_m", "quartile_performance.csv"))
        create_boxen_plot(quartile_performance, "MAE", save_path=save_path, model="ae_m")
    elif model == "GNN":
        quartile_performance = pd.read_csv(
            Path("data", "cleaned_data", "quartile_performance", "gnn", "quartile_performance.csv"))
        create_boxen_plot(quartile_performance, "MAE", save_path=save_path, model="gnn")
