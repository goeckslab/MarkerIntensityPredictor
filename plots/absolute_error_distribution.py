import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import plotly.graph_objects as go
from matplotlib.cbook import boxplot_stats

if __name__ == '__main__':
    # argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--errors", help="where to find the error scores")
    parser.add_argument("-m", "--markers", nargs='+')
    args = parser.parse_args()

    # load error scores
    errors = pd.read_csv(args.errors)

    # select only marker in args.markers
    if args.markers:
        errors = errors[args.markers]

    title = ""
    if "in_patient" in str(args.errors):
        title = "In patient"
    else:
        title = "Out patient"

    biopsy_name = Path(args.errors).stem.replace("_", " ")

    title = f"Biopsy {biopsy_name}\nLudwig {title}\nDistribution of the absolute error per cell"

    # Calculate outliers
    percent_outliers = []
    outlier_df = []
    for marker in errors.columns:
        # extract all outliers
        outliers = [y for stat in boxplot_stats(errors[marker]) for y in stat['fliers']]
        rows = errors[errors[marker].isin(outliers)][[marker]]
        outlier_df.append(rows)
        # extract all rows which matches the value of the outliers

        percent_outliers.append({
            "Marker": marker,
            "Percentage": len(outliers) / len(errors[marker]) * 100,
            "Total Outliers": len(outliers),
            "Total cells": len(errors[marker])
        })

    percent_outliers = pd.DataFrame(percent_outliers)
    outlier_df = pd.concat(outlier_df, axis=0)

    print(percent_outliers)

    fig = plt.figure(dpi=200, figsize=(10, 5))
    ax = sns.violinplot(data=errors, inner='box')
    sns.stripplot(data=outlier_df, jitter=True)
    labels = ax.get_xticklabels()

    labels = [
        f"{x.get_text()}\n% Outliers: {percent_outliers[percent_outliers['Marker'] == x.get_text()]['Percentage'].values[0]:.2f}%\nTotal Outliers: {percent_outliers[percent_outliers['Marker'] == x.get_text()]['Total Outliers'].values[0]}"
        for x in labels]

    ax.set_xticklabels(labels)
    plt.title(title)
    plt.ylabel("Absolute Error")
    plt.xlabel("Marker")
    plt.tight_layout()
    plt.savefig(f"{Path(args.errors).parent}/{biopsy_name.replace(' ', '_')}.png")

    print(errors)
