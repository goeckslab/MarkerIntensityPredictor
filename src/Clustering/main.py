import phenograph
import pandas as pd
from pathlib import Path
import plotly.express as px
import umap
import seaborn as sns
from sklearn import metrics
import numpy as np


class Clustering:
    results_folder = Path("results", "clusters")
    silhouette_scores = pd.DataFrame(columns=['name', 'score', "group"])
    calinski_harabasz_scores = pd.DataFrame(columns=['name', 'score', "group"])
    args = None

    def __init__(self, args):
        self.args = args

    def create_cluster(self):

        j: int = 0
        k: int = 0
        for file in self.args.files:
            data = pd.read_csv(file)
            name = self.args.names[k]

            communities, graph, Q = phenograph.cluster(data)
            clusters = pd.Series(communities)
            clusters.to_csv(f'{self.results_folder}/{name}_pg_clusters.csv')
            pg_clusters = pd.Series(communities)

            fit = umap.UMAP()

            latent_umap = fit.fit_transform(data)

            plot = sns.scatterplot(data=latent_umap, x=latent_umap[:, 0], y=latent_umap[:, 1], hue=pg_clusters)
            fig = plot.get_figure()
            fig.savefig(f"{self.results_folder}/{name}_pg_clusters.png")

            means = []
            for i in range(0, pd.Series(communities).max() + 1):
                cells_index = pg_clusters[pg_clusters == i].index
                filtered_markers_df = data[data.index.isin(cells_index)]
                means.append(filtered_markers_df.mean().values)

            fig = px.imshow(means, x=data.columns)
            # fig.show()

            self.silhouette_scores = self.silhouette_scores.append({
                "group": self.resolve_group(j),
                "name": name,
                "score": metrics.silhouette_score(data, clusters, metric='euclidean'),
                "method": "silhouette",
            },
                ignore_index=True)

            self.calinski_harabasz_scores = self.calinski_harabasz_scores.append({
                "group": self.resolve_group(j),
                "name": name,
                "score": metrics.calinski_harabasz_score(data, clusters),
                "method": "calinski_harabasz",
            },
                ignore_index=True)

            print(self.silhouette_scores)
            print(self.calinski_harabasz_scores)

            j += 1

            if k == 2:
                k = 0
            else:
                k += 1

        self.silhouette_scores.to_csv(f"{self.results_folder}/silhouette_scores.csv", index=False)
        self.calinski_harabasz_scores.to_csv(f"{self.results_folder}/calinski_harabasz_scores.csv", index=False)

        g = sns.catplot(
            data=self.silhouette_scores, kind="bar",
            x="group", y="score", hue="name",
            ci="sd", palette="dark", alpha=.6, height=6
        )
        g.despine(left=True)
        g.set_axis_labels("", "Cluster scores")
        g.fig.suptitle("Silhouette")

        g.savefig(Path(f"{self.results_folder}/silhouette_scores.png"))

        g = sns.catplot(
            data=self.calinski_harabasz_scores, kind="bar",
            x="group", y="score", hue="name",
            ci="sd", palette="dark", alpha=.6, height=6
        )
        g.despine(left=True)
        g.set_axis_labels("", "Cluster scores")
        g.fig.suptitle("Calinski Harabasz")

        g.savefig(Path(f"{self.results_folder}/calinski_harabasz_scores.png"))

    def resolve_group(self, i: int) -> str:
        if i <= 2:
            return "9_2_1"

        elif i <= 5:
            return "9_2_2"

        elif i <= 8:
            return "9_3_1"

        elif i <= 11:
            return "9_3_2"
