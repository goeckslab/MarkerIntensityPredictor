import os
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from statannotations.Annotator import Annotator

if __name__ == '__main__':
    data = []
    for subdir, dirs, files in os.walk(Path("en", "exp")):
        for file in files:
            if file == "results_top_3.csv":
                print(os.path.join(subdir, file))
                data.append(pd.read_csv(os.path.join(subdir, file), sep=",", header=0))

    for subdir, dirs, files in os.walk(Path("en", "ip")):
        for file in files:
            if file == "results_top_3.csv":
                print(os.path.join(subdir, file))
                data.append(pd.read_csv(os.path.join(subdir, file), sep=",", header=0))

    data = pd.concat(data, axis=0)
    #duplicate rows 30x for each markers
    data = pd.concat([data] * 30, ignore_index=True)
    print(data)

    temp = pd.DataFrame()
    # merge mean_absolute_error and target_mae_full panel columns
    temp["MAE"] = data["MAE"]
    temp["Marker"] = data["Target"]
    temp["Mode"] = data["Mode"]
    temp["Full Panel"] = 0

    temp_2 = pd.DataFrame()
    temp_2["MAE"] = data["MAE_FP"]
    temp_2["Marker"] = data["Target"]
    temp_2["Mode"] = data["Mode"]
    temp_2["Full Panel"] = 1

    temp = pd.concat([temp, temp_2], axis=0)
    # duplicate rows 30x for each markers
    # temp = pd.concat([temp] * 30, ignore_index=True)

    for mode in temp["Mode"].unique():
        data = temp[temp["Mode"] == mode]
        fig = plt.figure(figsize=(10, 5), dpi=200)
        ax = sns.boxenplot(x="Marker", y="MAE", hue="Full Panel", data=data, palette="Set3")
        hue = "Full Panel"
        hue_order = [1, 0]
        pairs = [
            (("pRB", 1), ("pRB", 0)),
            (("CD45", 1), ("CD45", 0)),
            (("CK19", 1), ("CK19", 0)),
            (("Ki67", 1), ("Ki67", 0)),
            (("aSMA", 1), ("aSMA", 0)),
            (("Ecad", 1), ("Ecad", 0)),
            (("PR", 1), ("PR", 0)),
            (("CK14", 1), ("CK14", 0)),
            (("HER2", 1), ("HER2", 0)),
            (("AR", 1), ("AR", 0)),
            (("CK17", 1), ("CK17", 0)),
            (("p21", 1), ("p21", 0)),
            (("Vimentin", 1), ("Vimentin", 0)),
            (("pERK", 1), ("pERK", 0)),
            (("EGFR", 1), ("EGFR", 0)),
            (("ER", 1), ("ER", 0)),
        ]
        order = ['pRB', 'CD45', 'CK19', 'Ki67', 'aSMA', 'Ecad', 'PR', 'CK14', 'HER2', 'AR', 'CK17', 'p21', 'Vimentin',
                 'pERK', 'EGFR', 'ER']
        annotator = Annotator(ax, pairs, data=temp, x="Marker", y="MAE", order=order, hue=hue, hue_order=hue_order,
                              verbose=1)
        annotator.configure(test='Mann-Whitney', text_format='star', loc='outside')
        annotator.apply_and_annotate()
        plt.legend(loc="upper left", title=f"Full Panel {mode.upper()}")
        plt.tight_layout()
        plt.savefig(Path("en", f"{mode}_performance.png"), dpi=200)
