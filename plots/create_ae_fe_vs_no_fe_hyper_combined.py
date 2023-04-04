import argparse, os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ip_non_fe_data = Path("ae/ip")
ip_fe_data = Path("ae_sp_46/ip")

op_non_fe_data = Path("ae/op")
op_fe_data = Path("ae_sp_46/op")

ip_save_path = Path("ip_plots/ae/")
op_save_path = Path("op_plots/ae/")


# Create line & violin plot for hyper & non hyper data + fe & non fe data.


def load_scores(source: Path) -> pd.DataFrame:
    scores = []

    for root, dirs, files in os.walk(source):
        for file in files:
            if file == "scores.csv" or file == "hp_scores.csv":
                print("Loading file: ", file)
                scores.append(pd.read_csv(os.path.join(root, file), sep=",", header=0))

    assert len(scores) == 16, f"Not all biopsies have been selected. Only {len(scores)} biopsies have been selected."

    return pd.concat(scores, axis=0).sort_values(by=["Marker"]).reset_index(drop=True)


def create_violin_plot(data: pd.DataFrame, metric: str, save_folder: Path, file_name: str,
                       ylim: tuple):
    data["Biopsy"] = data["Biopsy"].apply(lambda x: f"{x.replace('_', ' ')}").values
    if args.markers:
        fig = plt.figure(figsize=(13, 5), dpi=200)
    else:
        fig = plt.figure(figsize=(15, 5), dpi=200)
    ax = sns.violinplot(data=data, x="Marker", y=metric, hue="FE", split=True, cut=0)

    # plt.title(title)
    # remove y axis label
    # plt.ylabel("")
    # plt.xlabel("")
    # plt.legend(loc='upper center')
    plt.ylim(ylim[0], ylim[1])

    y_ticks = [item.get_text() for item in fig.axes[0].get_yticklabels()]
    x_ticks = [item.get_text() for item in fig.axes[0].get_xticklabels()]
    # set y ticks of fig
    if args.markers:
        ax.set_yticklabels(y_ticks, rotation=0, fontsize=20)
        ax.set_xticklabels(x_ticks, rotation=0, fontsize=20)
    plt.box(False)
    # remove legend from fig
    # plt.legend().set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{save_folder}/{file_name}.png")
    plt.close('all')


def create_line_plot(data: pd.DataFrame, metric: str, save_folder: Path, file_name: str):
    fig = plt.figure(figsize=(10, 5), dpi=200)
    ax = sns.lineplot(x="Marker", y=metric, hue="FE", style="HP", data=data)
    plt.tight_layout()
    plt.savefig(f"{save_folder}/{file_name}.png")
    plt.close('all')


def combine_line_violin_plot(data: pd.DataFrame, metric: str, save_folder: Path, file_name: str):
    fig = plt.figure(figsize=(10, 5), dpi=200)
    ax = sns.lineplot(x="Marker", y=metric, hue="FE", style="HP", data=data)
    sns.violinplot(data=data, x="Marker", y=metric, hue="FE", split=True, cut=0)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--markers", nargs='+', )
    parser.add_argument("-t", "--type", type=str, choices=["ip", "op"], default="ip")
    parser.add_argument("-s", "--scores", type=str, default="MAE")
    args = parser.parse_args()

    patient_type = args.type
    metric: str = args.scores

    if patient_type == "ip":
        save_path = ip_save_path
        fe_data = ip_fe_data
        non_fe_data = ip_non_fe_data
    else:
        save_path = op_save_path
        fe_data = op_fe_data
        non_fe_data = op_non_fe_data

    if not save_path.exists():
        save_path.mkdir(parents=True)

    non_fe_scores = load_scores(non_fe_data)
    non_fe_scores["FE"] = "sp_0"

    fe_scores = load_scores(fe_data)

    scores = pd.concat([non_fe_scores, fe_scores], axis=0).reset_index(drop=True)

    if args.markers:
        scores = scores[scores["Marker"].isin(args.markers)]

    if args.markers:
        violin_file_name: str = f"denoising_fe_vs_non_fe_{metric.lower()}_violin_{patient_type}_combined_hp"
        line_file_name: str = f"denoising_fe_vs_non_fe_{metric.lower()}_line_{patient_type}_combined_hp"
    else:
        violin_file_name: str = f"denoising_fe_vs_non_fe_{metric.lower()}_violin_{patient_type}_combined_hp_all_markers"
        line_file_name: str = f"denoising_fe_vs_non_fe_{metric.lower()}_line_{patient_type}_combined_hp_all_markers"

    create_violin_plot(data=scores, metric=metric, save_folder=save_path, file_name=violin_file_name, ylim=(0, 0.5))

    create_line_plot(data=scores, metric=metric.upper(), file_name=line_file_name,
                     save_folder=save_path)

    # combine_line_violin_plot(data=scores, metric=metric.upper(), save_folder=save_path,
    #                         file_name=f"denoising_fe_vs_non_fe_{metric.lower()}_line_violin_{patient_type}")
