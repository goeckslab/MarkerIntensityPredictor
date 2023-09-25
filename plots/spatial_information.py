import pandas as pd
import logging, sys
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

SHARED_MARKERS = ['pRB', 'CD45', 'CK19', 'Ki67', 'aSMA', 'Ecad', 'PR', 'CK14', 'HER2', 'AR', 'CK17', 'p21', 'Vimentin',
                  'pERK', 'EGFR', 'ER']

logging.root.handlers = []
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("plots/debug.log"),
                        logging.StreamHandler()
                    ])


save_path = Path("data", "cleaned_data", "spatial_information")

if __name__ == '__main__':
    spatial_information = pd.read_csv("data/cleaned_data/spatial_information/spatial_clustering.csv")
    scores = pd.read_csv("data/cleaned_data/scores/lgbm/scores.csv")

    # select only non hp scores
    scores = scores[scores["HP"] == 0]
    # select only FE which is not 0
    scores = scores[scores["FE"] != 0]
    # select only EXP scores
    scores = scores[scores["Mode"] == "EXP"]

    # calculate the mean MAE for each marker, biopsy and FE
    mean_mae: pd.DataFrame = scores.groupby(["Marker", "Biopsy", "FE"])["MAE"].mean().reset_index()

    errors = []
    for biopsy in mean_mae["Biopsy"].unique():
        for marker in mean_mae["Marker"].unique():
            for fe in mean_mae["FE"].unique():
                # get the mean MAE for the current marker, biopsy and FE
                try:
                    mean_mae_value = mean_mae[(mean_mae["Biopsy"] == biopsy) & (mean_mae["Marker"] == marker) & (
                            mean_mae["FE"] == fe)]["MAE"].values[0]

                except BaseException as ex:
                    errors.append({"Biopsy": biopsy, "Marker": marker, "FE": fe})

    assert len(errors) == 0, "There are missing values in the mean MAE dataframe"

    # replace FE values for mean_mae with the actual FE values
    mean_mae["FE"] = mean_mae["FE"].replace({23: 15, 46: 30, 92: 60, 138: 90, 184: 120})
    # replace _ in biopsy with ' '
    spatial_information["Biopsy"] = spatial_information["Biopsy"].str.replace("_", " ")

    # merge the I column from the spatial information with the mean MAE per marker, biopsy and FE
    merged_information: pd.DataFrame = spatial_information.merge(mean_mae, on=["Biopsy", "Marker", "FE"], how="left")
    # save merged information
    merged_information.to_csv(Path(save_path, "clustering_and_performance.csv"), index=False)



    # create scatterplot using marker, biopsy, I and MAE
    sns.scatterplot(data=merged_information, x="I", y="MAE", hue="Marker")
    plt.show()
    sys.exit()

    print(merged_information.head(20))
    # select only MAE for marker CK19
    temp = merged_information[merged_information["Marker"] == "CK19"]
    print(temp)
    input()

    # calculate correlation between I and MAE for each marker and each biopsy
    correlations = []
    for biopsy in merged_information["Biopsy"].unique():
        sub_df = merged_information[merged_information["Biopsy"] == biopsy]
        for marker in sub_df["Marker"].unique():
            sub_sub_df = sub_df[sub_df["Marker"] == marker]
            correlation = sub_sub_df["I"].corr(sub_sub_df["MAE"])
            correlations.append({"Biopsy": biopsy, "Marker": marker, "Correlation": correlation})

    biopsy_correlations = pd.DataFrame(correlations)
    print(biopsy_correlations)

    # plot correlation heatmap
    # biopsy_correlations = biopsy_correlations.pivot(index="Biopsy", columns="Marker", values="Correlation")
    # sns.heatmap(biopsy_correlations, annot=True, cmap="coolwarm")
    # plt.show()

    # calculate correlation between I and MAE for each marker
    correlations = []
    for marker in merged_information["Marker"].unique():
        sub_df = merged_information[merged_information["Marker"] == marker]
        correlation = sub_df["I"].corr(sub_df["MAE"])
        correlations.append({"Marker": marker, "Correlation": correlation})

    marker_correlations = pd.DataFrame(correlations)
    print(marker_correlations)

    # plot correlation heatmap
    # marker_correlations = marker_correlations.pivot(index="Marker", columns="Marker", values="Correlation")
    # sns.heatmap(marker_correlations, annot=True, cmap="coolwarm")
    # plt.show()

    # calculate correlation between I and MAE for for each FE per marker
    correlations = []
    for fe in merged_information["FE"].unique():
        sub_df = merged_information[merged_information["FE"] == fe]
        for marker in sub_df["Marker"].unique():
            sub_sub_df = sub_df[sub_df["Marker"] == marker]
            correlation = sub_sub_df["I"].corr(sub_sub_df["MAE"])
            correlations.append({"FE": fe, "Marker": marker, "Correlation": correlation})

    fe_correlations = pd.DataFrame(correlations)
    print(fe_correlations)

    # plot correlation heatmap
    fe_correlations = fe_correlations.pivot(index="FE", columns="FE", values="Correlation")
    sns.heatmap(fe_correlations, annot=True, cmap="coolwarm")
    plt.show()


    # plot mae per marker