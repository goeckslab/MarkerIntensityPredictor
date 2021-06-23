import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def __create_intensity_heatmap_plot(self):
    fig, ax = plt.subplots(figsize=(30, 30), dpi=300)  # Sample figsize in inches
    sns.heatmap(self.train_data.X_train, xticklabels=self.train_data.markers)
    ax.set_title("Marker intensities")
    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig(Path(f"results/lr/marker_heatmap.png"), bbox_inches='tight')
    plt.close()


def __create_r2_accuracy_plot(self):
    """
    Creates a bar plot showing the accuracy of the model for each marker
    :return:
    """
    ax = sns.catplot(
        data=self.prediction_scores, kind="bar",
        x="Score", y="Marker", hue="Model",
        ci="sd", palette="dark", alpha=.6, height=6
    )
    ax.despine(left=True)
    ax.set_axis_labels("R2 Score", "Marker")
    ax.set(xlim=(0, 1))

    if self.test_file is None:
        # ax.fig.suptitle("Single file")
        plt.title("Single File", y=1.02)
        ax.legend.set_title("Model")
        ax.savefig(Path(f"results/lr/{self.train_file_name}_score_predictions.png"))
    elif self.train_file is None:
        plt.title("Multi Files", y=1.02)
        ax.legend.set_title("Model")
        ax.savefig(Path(f"results/lr/{self.test_file_name}_multi_score_predictions.png"))
    else:
        plt.title("Train Test File", y=1.02)
        ax.legend.set_title("Model")
        ax.savefig(Path(f"results/lr/{self.train_file_name}_{self.test_file_name}_score_predictions.png"))

    plt.close()
