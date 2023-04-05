import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.cbook import boxplot_stats

ip_path = Path("ip_plots/")
op_path = Path("op_plots/")

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
    in_patient = False
    biopsy_name = Path(args.errors).stem.replace("_", " ")

    file_name = f"{biopsy_name.replace(' ', '_')}_error_distribution"
    if "in_patient" in str(args.errors):
        title = "In patient"
        in_patient = True
        save_path = ip_path
    else:
        title = "Out patient"
        save_path = op_path

    if "_en" in str(args.errors):
        title += " EN"
        save_path = Path(save_path / "en")
    elif "_hyper" in str(args.errors):
        title += " Ludwig Hyper"
        save_path = Path(save_path / "ludwig_hyper")
    elif "_sp_" in str(args.errors):
        title += " Ludwig Spatial"
        save_path = Path(save_path / "ludwig_fe")
        raise ValueError("Spatial resolution not implemented yet")
        file_name += "_sp_"
    else:
        title += " Ludwig"
        save_path = Path(save_path / "ludwig")

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
        f"{x.get_text()}\nOutliers: {percent_outliers[percent_outliers['Marker'] == x.get_text()]['Total Outliers'].values[0]} " \
        f"({percent_outliers[percent_outliers['Marker'] == x.get_text()]['Percentage'].values[0]:.1f}%)"
        for x in labels]

    ax.set_xticklabels(labels)
    plt.title(title)
    plt.ylabel("Absolute Error")
    plt.xlabel("Marker")
    plt.tight_layout()

    print(save_path)
    print(file_name)
    if in_patient:
        plt.savefig(f"{save_path}/{file_name}.png")
    else:
        plt.savefig(f"{save_path}/{file_name}.png")

    print(errors)
