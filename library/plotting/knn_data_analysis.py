from pathlib import Path
from typing import List, Dict
import mlflow
import matplotlib.pyplot as plt
import seaborn as sns


class KNNDataAnalysisPlotting:

    def __init__(self, base_path: Path):
        self._base_path = base_path

    def pie_chart(self, keys: List, data: List, file_name: str, title: str, mlflow_directory: str = None,
                  colors: Dict = None):
        # define Seaborn color palette to use

        if colors is None:
            color_map = sns.color_palette('bright')
        else:
            color_map = [colors[v] for v in keys]

        # plotting data on chart
        plt.pie(data, labels=keys, colors=color_map, autopct='%.0f%%')
        plt.title(title)
        plt.tight_layout()
        save_path = Path(self._base_path, f"{file_name}.png")
        plt.savefig(save_path)

        if mlflow_directory is not None:
            mlflow.log_artifact(str(save_path), mlflow_directory)
        else:
            mlflow.log_artifact(str(save_path))
        plt.close()
