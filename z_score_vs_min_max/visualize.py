import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statannotations.Annotator import Annotator

if __name__ == '__main__':
    df = pd.read_csv("z_score_vs_min_max/scores.csv")
    # rename Scaler to Scaling
    df = df.rename(columns={"Scaler": "Scaling"})
    # rename Standard to Z-Score
    df = df.replace("Standard", "Z-Score")

    # change RMSEPE to % Error
    df = df.rename(columns={"RMSEPE": "% Error"})

    # divide $% Error by 10
    df["% Error"] = df["% Error"] / 10

    # rename marker to Protein
    df = df.rename(columns={"Marker": "Protein"})

    # remove all outliers which are greater than 3 standard deviations from the mean
    df = df[df["% Error"] < df["% Error"].mean() + 3 * df["% Error"].std()]
    # sort df by marker
    df = df.sort_values(by=["Protein"])

    hue = "Scaling"
    hue_order = ["Z-Score", "Min Max"]
    y = "% Error"
    x = "Protein"

    fig, ax = plt.subplots(figsize=(15, 10))
    sns.boxenplot(x=x, y=y, hue=hue, data=df)

    pairs = [
        (("AR", "Z-Score"), ("AR", "Min Max")),
        (("CK14", "Z-Score"), ("CK14", "Min Max")),
        (("CK17", "Z-Score"), ("CK17", "Min Max")),
        (("CK19", "Z-Score"), ("CK19", "Min Max")),
        (("CD45", "Z-Score"), ("CD45", "Min Max")),
        (("Ecad", "Z-Score"), ("Ecad", "Min Max")),
        (("EGFR", "Z-Score"), ("EGFR", "Min Max")),
        (("ER", "Z-Score"), ("ER", "Min Max")),
        (("HER2", "Z-Score"), ("HER2", "Min Max")),
        (("Ki67", "Z-Score"), ("Ki67", "Min Max")),
        (("PR", "Z-Score"), ("PR", "Min Max")),
        (("Vimentin", "Z-Score"), ("Vimentin", "Min Max")),
        (("aSMA", "Z-Score"), ("aSMA", "Min Max")),
        (("p21", "Z-Score"), ("p21", "Min Max")),
        (("pERK", "Z-Score"), ("pERK", "Min Max")),
        (("pRB", "Z-Score"), ("pRB", "Min Max")),

    ]

    order = ["AR", "CK14", "CK17", "CK19", "CD45", "Ecad", "EGFR", "ER", "HER2", "Ki67", "PR", "Vimentin", "aSMA",
             "p21", "pERK", "pRB"]
    annotator = Annotator(ax, pairs, data=df, x=x, y=y, order=order, hue=hue, hue_order=hue_order,
                          verbose=1)
    annotator.configure(test='Mann-Whitney', text_format='star', loc='outside')
    annotator.apply_and_annotate()

    # plt.ylim(-40, 1000)
    # change title, increase font size to 20 and move it up
    plt.title("Z-Score vs Min Max Scaling", fontsize=20, y=1.10)
    plt.tight_layout()
    plt.savefig("z_score_vs_min_max/percent_wise_errors.png", dpi=300)
    plt.close()
