from scipy.stats import ranksums
import pandas as pd, os, numpy as np, matplotlib.pyplot as plt, seaborn as sns
from pathlib import Path
from scipy import stats

results_folder = Path("stat_plots")

if not results_folder.exists():
    results_folder.mkdir(parents=True, exist_ok=True)


def rank_biopsy_performance(df: pd.DataFrame, title: str, save_name: str):
    df["Biopsy"] = [biopsy.replace('_', ' ') for biopsy in list(df["Biopsy"].values)]

    df = df.pivot(index="Biopsy", columns="Marker", values="Score")

    ranks = []
    # RAnk each marker values and replace the value with biopsy name
    for marker in df.columns:
        data = df[marker].rank(ascending=True)
        ranks.append(data)

    data = pd.concat(ranks, axis=1)

    fig = plt.figure(figsize=(15, 5), dpi=200)
    sns.heatmap(data, annot=True, cmap="YlGnBu")
    plt.title(f"{title}\nHigher value indicate lower performance")
    for ax in fig.axes:
        for tick in ax.get_yticklabels():
            tick.set_rotation(0)
    plt.tight_layout()
    plt.savefig(f"{results_folder}/{save_name}.png")

    return data


if __name__ == '__main__':

    biopsies = ["9 2 1", "9 2 2", "9 3 1", "9 3 2", "9 14 1", "9 14 2", "9 15 1", "9 15 2"]
    # load mesmer mae scores from data mesmer folder and all subfolders

    mae_scores = []
    for root, dirs, files in os.walk("mesmer"):
        for name in files:
            if Path(name).suffix == ".csv" and "_mae_scores" in name:
                mae_scores.append(pd.read_csv(os.path.join(root, name), sep=",", header=0))

    for root, dirs, files in os.walk("unmicst_s3_non_snr"):
        for name in files:
            if Path(name).suffix == ".csv" and "_mae_scores" in name:
                mae_scores.append(pd.read_csv(os.path.join(root, name), sep=",", header=0))

    for root, dirs, files in os.walk("unmicst_s3_snr"):
        for name in files:
            if Path(name).suffix == ".csv" and "_mae_scores" in name:
                mae_scores.append(pd.read_csv(os.path.join(root, name), sep=",", header=0))

    assert len(mae_scores) == 48, "There should be 48 mae scores files but only {} were found".format(len(mae_scores))
    mae_scores = pd.concat(mae_scores, axis=0).reset_index(drop=True)

    # Rank biopsy performance
    mesmer_op_ranked = rank_biopsy_performance(
        mae_scores[(mae_scores["Type"] == "OP") & (mae_scores["Segmentation"] == "Mesmer")],
        title="Mesmer OP Biopsy Ranking",
        save_name="mesmer_op")

    mesmer_ip_ranked = rank_biopsy_performance(
        mae_scores[(mae_scores["Type"] == "IP") & (mae_scores["Segmentation"] == "Mesmer")],
        title="Mesmer IP Biopsy Ranking",
        save_name="mesmer_ip")

    s3_snr_ip_ranked = rank_biopsy_performance(
        mae_scores[
            (mae_scores["Type"] == "IP") & (mae_scores["Segmentation"] == "Unmicst + S3") & (mae_scores["SNR"] == 1)],
        title="UnMICST + S3 SNR IP Biopsy Ranking",
        save_name="s3_snr_mesmer_ip")

    s3_snr_op_ranked = rank_biopsy_performance(
        mae_scores[
            (mae_scores["Type"] == "OP") & (mae_scores["Segmentation"] == "Unmicst + S3") & (mae_scores["SNR"] == 1)],
        title="UnMICST + S3 SNR OP Biopsy Ranking",
        save_name="s3_snr_mesmer_Op")

    s3_ip_ranked = rank_biopsy_performance(
        mae_scores[
            (mae_scores["Type"] == "IP") & (mae_scores["Segmentation"] == "Unmicst + S3") & (mae_scores["SNR"] == 0)],
        title="UnMICST + S3 IP Biopsy Ranking",
        save_name="s3_mesmer_ip")

    s3_op_ranked = rank_biopsy_performance(
        mae_scores[
            (mae_scores["Type"] == "OP") & (mae_scores["Segmentation"] == "Unmicst + S3") & (mae_scores["SNR"] == 0)],
        title="UnMICST + S3 OP Biopsy Ranking",
        save_name="s3_mesmer_op")

    # Extract out patient scores from mae_scores
    op = mae_scores[mae_scores["Type"] == "OP"].copy()

    # Extract memser scores from op
    op_mesmer = op[op["Segmentation"] == "Mesmer"].copy()
    # Extract Unmicst scores and snr from op
    op_unmicst_s3_snr = op[(op["Segmentation"] == "Unmicst + S3") & (op["SNR"] == 1)].copy()
    op_unmicst_s3_non_snr = op[(op["Segmentation"] == "Unmicst + S3") & (op["SNR"] == 0)].copy()

    p_values = []
    for biopsy in biopsies:
        biopsy = biopsy.replace(" ", "_")
        mesmer_score = op_mesmer[op_mesmer["Biopsy"] == biopsy]["Score"].values
        non_snr_score = op_unmicst_s3_non_snr[op_unmicst_s3_non_snr["Biopsy"] == biopsy]["Score"].values
        snr_score = op_unmicst_s3_snr[op_unmicst_s3_snr["Biopsy"] == biopsy]["Score"].values

        p_values.append(
            {
                "Biopsy": biopsy,
                "Alternative": "Two sided",
                "Mesmer vs Non SNR": ranksums(mesmer_score, non_snr_score)[1],
                "Mesmer vs SNR": ranksums(mesmer_score, snr_score)[1],
                "Non SNR vs SNR": ranksums(non_snr_score, snr_score)[1],
            }
        )

        p_values.append(
            {
                "Biopsy": biopsy,
                "Alternative": "Less",
                "Mesmer vs Non SNR": ranksums(mesmer_score, non_snr_score, alternative="less")[1],
                "Mesmer vs SNR": ranksums(mesmer_score, snr_score, alternative="less")[1],
                "Non SNR vs SNR": ranksums(non_snr_score, snr_score, alternative="less")[1],
            }
        )

        p_values.append(
            {
                "Biopsy": biopsy,
                "Alternative": "Greater",
                "Mesmer vs Non SNR": ranksums(mesmer_score, non_snr_score, alternative="greater")[1],
                "Mesmer vs SNR": ranksums(mesmer_score, snr_score, alternative="greater")[1],
                "Non SNR vs SNR": ranksums(non_snr_score, snr_score, alternative="greater")[1],
            }
        )

    # p_values.append({
    #    "Biopsy": biopsy,
    #    "Alternative": "Two sided",
    #    "Mesmer vs Non SNR": stats.kstest(mesmer_score, non_snr_score)[1],
    #    "Mesmer vs SNR": stats.kstest(mesmer_score, snr_score)[1],
    #    "Non SNR vs SNR": stats.kstest(non_snr_score, snr_score)[1],
    #
    #            }
    #        )

    p_values = pd.DataFrame(p_values)
    print(p_values)
