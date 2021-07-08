import seaborn as sns
from pathlib import Path
import matplotlib.pyplot as plt
import os
import logging
import numpy as np

sns.set_theme(style="darkgrid")
results_folder = Path("results", "plots")


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
    def plot_reconstructed_markers(X, X_pred, data_origin):
        logging.info("Plotting reconstructed intensities")
        markers = X.columns
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 10), dpi=300, sharex=True)
        sns.heatmap(X, ax=ax1, xticklabels=markers)
        sns.heatmap(X_pred, ax=ax2, xticklabels=markers)

        ax1.set_title("X Validation")
        ax2.set_title("Reconstructed X Validation")
        fig.tight_layout()
        plt.savefig(Path(f"{results_folder}/{data_origin}_reconstructed.png"))
        plt.close()

    @staticmethod
    def plot_corr_heatmap(correlation_df, data_origin: str):
        plt.figure(figsize=(20, 16))
        # define the mask to set the values in the upper triangle to True
        mask = np.triu(np.ones_like(correlation_df, dtype=np.bool))
        heatmap = sns.heatmap(correlation_df, mask=mask, vmin=-1, vmax=.6, annot=True, cmap='BrBG')
        heatmap.set_title('Correlation Heatmap of Markers', fontdict={'fontsize': 18}, pad=16)

        heatmap.set_yticklabels(correlation_df.columns)

        for item in heatmap.get_yticklabels():
            item.set_rotation(0)

        plt.savefig(Path(f"{results_folder}/{data_origin}_correlation.png"))
        plt.close()

    @staticmethod
    def plot_combined_corr_plot(df):
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(40, 30), dpi=300)

        column = 0
        row = 0
        for file in df["File"].unique():
            correlation_df = df[df["File"] == file]
            del correlation_df["File"]

            mask = np.triu(np.ones_like(correlation_df, dtype=np.bool))
            heatmap = sns.heatmap(correlation_df, ax=axes[row, column], mask=mask, vmin=-1, vmax=.6, annot=True,
                                  cmap='BrBG')
            # heatmap.set_title('Correlation Heatmap of Markers', fontdict={'fontsize': 18}, pad=16)
            axes[row, column].set_title(file)
            heatmap.set_yticklabels(correlation_df.columns)

            for item in heatmap.get_yticklabels():
                item.set_rotation(0)

            column += 1
            # Reset column and increase row by 1
            if column == 2:
                column = 0
                row += 1

        plt.savefig(Path(f"{results_folder}/combined_correlation.png"))
        plt.close()

    @staticmethod
    def __create_linear_coefficients_heatmap_plot(self):
        """
        Creates a heatmap for the coefficients
        """

        for model in self.coefficients["Model"].unique():
            df = self.coefficients[self.coefficients["Model"] == model].copy()

            del df["Model"]
            df.fillna(-1, inplace=True)
            # df[(df < 0.001) | (df > 0.6)] = None
            fig, ax = plt.subplots(figsize=(20, 20))  # Sample figsize in inches
            ax = sns.heatmap(df, linewidths=1, vmin=0, vmax=0.6, cmap="YlGnBu", ax=ax)
            ax.set_title(model)
            fig = ax.get_figure()
            fig.savefig(Path(f"{self.results_folder}/{model}_coef_heatmap.png"), bbox_inches='tight')
            plt.close()
