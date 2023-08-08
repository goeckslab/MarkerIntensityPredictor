import argparse

import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", required=True)
    parser.add_argument("--title", "-t", required=True)
    parser.add_argument("--output", "-o", required=True)

    args = parser.parse_args()
    output: str = args.output
    input_file: str = args.input
    title: str = args.title

    df = pd.read_csv(Path(input_file))
    color_palette = {-1.0: "blue", 0.0: "red", 0.5: "orange", 0.9: "green"}
    # plot boxenplots
    sns.set_theme(style="whitegrid")
    sns.set_context("paper")
    sns.set(font_scale=1.5)
    fig, ax = plt.subplots(figsize=(15, 10))
    sns.scatterplot(x="Marker", y="MAE", data=df, ax=ax, hue="Correlation", palette=color_palette)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(Path("results", f"{output}.png"))
