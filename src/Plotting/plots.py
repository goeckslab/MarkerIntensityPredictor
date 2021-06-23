import seaborn as sns
from pathlib import Path
import matplotlib.pyplot as plt
import os
import logging

sns.set_theme(style="darkgrid")
results_folder = f"{os.path.split(os.environ['VIRTUAL_ENV'])[0]}/results/plots"


class Plots:
    @staticmethod
    def r2_scores_combined(r2_scores_df):
        g = sns.catplot(
            data=r2_scores_df, kind="bar",
            x="Score", y="Marker", hue="Model",
            ci="sd", palette="dark", alpha=.6,
            height=20, aspect=1
        )
        g.despine(left=True)
        g.set_axis_labels("R2 Score", "Marker")
        g.set(xlim=(0, 1))

        plt.title("R2 Scores", y=1.02)
        g.legend.set_title("Model")

        # extract the matplotlib axes_subplot objects from the FacetGrid
        # ax = g.facet_axis(0, 0)

        # iterate through the axes containers
        # for c in ax.containers:
        #    labels = [f'{(v.get_width() / 1000):.1f}' for v in c]
        #    ax.bar_label(c, labels=labels, label_type='edge')

        g.savefig(Path(f"{results_folder}/r2_scores.png"))

        plt.close()

    @staticmethod
    def plot_reconstructed_markers(X, X_pred, file_name):
        logging.info("Plotting reconstructed intensities")
        markers = X.columns
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 10), dpi=300, sharex=True)
        sns.heatmap(X, ax=ax1, xticklabels=markers)
        sns.heatmap(X_pred, ax=ax2, xticklabels=markers)

        ax1.set_title("X Validation")
        ax2.set_title("Reconstructed X Validation")
        fig.tight_layout()
        plt.savefig(Path(f"{results_folder}/{file_name}_reconstructed.png"))
        plt.close()
