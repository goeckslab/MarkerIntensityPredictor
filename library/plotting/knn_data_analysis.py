from pathlib import Path
from typing import List, Dict, Union
import mlflow
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


class KNNDataAnalysisPlotting:

    def __init__(self, base_path: Path):
        self._base_path = base_path

    def pie_chart(self, keys: List, data: Union[pd.Series, Dict], file_name: str, title: str = None,
                  mlflow_directory: str = None, colors: Dict = None, label_func=None):
        """

        @param keys:
        @param data:
        @param file_name:
        @param title:
        @param mlflow_directory:
        @param colors:
        @param label_func: A function to create the labels on top of the pieces
        @return:
        """

        if title is None:
            title = file_name

        # define Seaborn color palette to use
        if colors is None:
            color_map = sns.color_palette('bright')
        else:
            color_map = [colors[v] for v in keys]

        if isinstance(data, pd.Series):
            data = pd.Series(data).value_counts().to_list()
            # plotting data on chart
            plt.pie(data, labels=keys, colors=color_map, autopct='%.0f%%')

        else:
            n_cols = len(data.keys()) if len(data.keys()) <= 3 else 3
            n_rows = self.__calculate_n_rows(data=data)
            fig, axs = self.__get_fig_and_axs(n_cols=n_cols, n_rows=n_rows)

            row = 0
            col = 0
            for key in data.keys():
                value = data[key].value_counts().to_list()

                if n_rows == 1:
                    if n_cols == 1:
                        if label_func is None:
                            axs.pie(value, labels=keys, colors=color_map, autopct='%.0f%%')
                        else:
                            axs.pie(value, labels=keys, color=color_map, autopct=lambda pct: label_func(pct, data))
                        axs.set_title(f"Neighbo {key}")
                        axs.set_xticklabels(axs.get_xticklabels(), rotation=90)
                    else:
                        if label_func is None:
                            axs[col].pie(value, labels=keys, colors=color_map, autopct='%.0f%%')
                        else:
                            axs[col].pie(value, labels=keys, colors=color_map,
                                         autopct=lambda pct: label_func(pct, data))
                        axs[col].set_title(f"Neighbor {key}")
                        axs[col].set_xticklabels(axs[col].get_xticklabels(), rotation=90)

                else:
                    if label_func is None:
                        axs[row, col].pie(value, labels=keys, colors=color_map, autopct='%.0f%%')
                    else:
                        axs[row, col].pie(value, labels=keys, colors=color_map,
                                          autopct=lambda pct: label_func(pct, data))
                    axs[row, col].set_title(f"Neighbor {key}")
                    axs[row, col].set_xticklabels(axs[row, col].get_xticklabels(), rotation=90)

                if col == n_cols - 1:
                    row += 1
                    col = 0

                else:
                    col += 1

        plt.tight_layout()
        save_path = Path(self._base_path, f"{file_name}.png")
        plt.savefig(save_path)

        if mlflow_directory is not None:
            mlflow.log_artifact(str(save_path), mlflow_directory)
        else:
            mlflow.log_artifact(str(save_path))
        plt.close()

    def line_plot(self, data, x: str, y: str, title: str, file_name: str, hue: str = None,
                  mlflow_directory: str = None, style: str = None):

        if style is None:
            sns.lineplot(data=data, x=x, y=y, hue=hue)
        else:
            sns.lineplot(data=data, x=x, y=y, hue=hue, style=style)

        plt.title(title)
        plt.tight_layout()
        plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', ncol=1)
        save_path = Path(self._base_path, f"{file_name}.png")
        plt.savefig(save_path, bbox_inches='tight')

        if mlflow_directory is not None:
            mlflow.log_artifact(str(save_path), mlflow_directory)
        else:
            mlflow.log_artifact(str(save_path))
        plt.close()

    @staticmethod
    def __calculate_n_rows(data: Dict):
        num_rows = 1
        if len(data.items()) > 3:
            num_rows = float(len(data.items()) / 3)
            if not num_rows.is_integer():
                num_rows += 1

            num_rows = int(num_rows)

        return num_rows

    @staticmethod
    def __get_fig_and_axs(n_cols, n_rows: int):
        # Adjust figure size
        if n_rows == 1:
            fig, axs = plt.subplots(ncols=n_cols, nrows=n_rows, figsize=(10, 9), dpi=300, sharex=False)
        elif n_rows == 2:
            fig, axs = plt.subplots(ncols=n_cols, nrows=n_rows, figsize=(10, 9), dpi=300, sharex=False)
        elif n_rows == 3:
            fig, axs = plt.subplots(ncols=n_cols, nrows=n_rows, figsize=(10, 9), dpi=300, sharex=False)
        else:
            fig, axs = plt.subplots(ncols=n_cols, nrows=n_rows, figsize=(9, 7), dpi=300, sharex=False)

        return fig, axs
