import phenograph
import pandas as pd
from pathlib import Path
import plotly.express as px
import umap
import seaborn as sns
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


class ClusterAnalysis:
    files = None
    results_folder = Path("results", "cluster_analysis")
    cluster_assignments = pd.DataFrame(columns=['pca', 'vae', 'non'])
    silhouette_scores = pd.DataFrame(columns=['model', 'score', "group", "cluster"])
    calinski_harabasz_scores = pd.DataFrame(columns=['model', 'score', "group", "cluster"])
    args = None

    def __init__(self, args):
        self.args = args
        self.files = self.args.files

    def create_mean_score_plots(self):

        frames = []
        for file in self.files:
            data = pd.read_csv(file)
            if 'silhouette' in file.name:
                data['algo'] = "silhouette"

            elif 'calinski' in file.name:
                data['algo'] = "calinski"
            else:
                print("invalid file, skipping")
                continue
            frames.append(data)

        scores = pd.concat(frames).reset_index()

        for algo in scores["algo"].unique():
            data = scores[scores["algo"] == algo]
            mean_data = pd.DataFrame(columns=["model", "score", "cluster"])
            for cluster in data["cluster"].unique():
                clusters = data[data["cluster"] == cluster]
                for model in clusters["model"].unique():
                    mean_data = mean_data.append(
                        {
                            "model": model,
                            "score": clusters[clusters["model"] == model]["score"].mean(),
                            "cluster": cluster,
                        }, ignore_index=True)

            g = sns.catplot(
                data=mean_data, kind="bar",
                x="cluster", y="score", hue="model",
                ci="sd", palette="dark", alpha=.6, height=6
            )
            g.despine(left=True)
            g.set_axis_labels("", "Cluster scores")
            g.fig.suptitle("Silhouette")

            g.savefig(Path(f"{self.results_folder}/{algo}_mean_scores.png"))
            plt.close('all')

    def create_cluster(self):
        j: int = 0
        k: int = 0

        for file in self.files:
            data = pd.read_csv(file)
            name = self.args.names[k]
            group = self.__resolve_group(j)

            silhouette, calinksi = self.__create_k_means_cluster(data, name)

            self.silhouette_scores = self.silhouette_scores.append({
                "group": group,
                "model": name,
                "score": silhouette,
                "cluster": "kmeans"
            },
                ignore_index=True)

            self.calinski_harabasz_scores = self.calinski_harabasz_scores.append({
                "group": group,
                "model": name,
                "score": calinksi,
                "cluster": "kmeans"
            },
                ignore_index=True)

            silhouette, calinksi = self.__create_phenograph_cluster(data, name)

            self.silhouette_scores = self.silhouette_scores.append({
                "group": group,
                "model": name,
                "score": silhouette,
                "cluster": "pg"
            },
                ignore_index=True)

            self.calinski_harabasz_scores = self.calinski_harabasz_scores.append({
                "group": group,
                "model": name,
                "score": calinksi,
                "cluster": "pg"
            },
                ignore_index=True)

            j += 1

            if k == 2:
                k = 0
            else:
                k += 1

            plt.close('all')

        self.silhouette_scores.to_csv(f"{self.results_folder}/silhouette_scores.csv", index=False)
        self.calinski_harabasz_scores.to_csv(f"{self.results_folder}/calinski_harabasz_scores.csv", index=False)

        for cluster in self.silhouette_scores["cluster"].unique():
            data = self.silhouette_scores[self.silhouette_scores["cluster"] == cluster]
            g = sns.catplot(
                data=data, kind="bar",
                x="group", y="score", hue="model",
                ci="sd", palette="dark", alpha=.6, height=6
            )
            g.despine(left=True)
            g.set_axis_labels("", "Cluster scores")
            g.fig.suptitle("Silhouette")

            g.savefig(Path(f"{self.results_folder}/{cluster}_silhouette_scores.png"))
            plt.close('all')

        for cluster in self.calinski_harabasz_scores["cluster"].unique():
            data = self.calinski_harabasz_scores[self.calinski_harabasz_scores["cluster"] == cluster]
            g = sns.catplot(
                data=data, kind="bar",
                x="group", y="score", hue="model",
                ci="sd", palette="dark", alpha=.6, height=6
            )
            g.despine(left=True)
            g.set_axis_labels("", "Cluster scores")
            g.fig.suptitle("Calinski Harabasz")

            g.savefig(Path(f"{self.results_folder}/{cluster}_calinski_harabasz_scores.png"))
            plt.close('all')

    def __create_k_means_cluster(self, data, name) -> (int, int):
        print("Creating kmeans clusters ...")

        model = KMeans(n_clusters=3)
        model.fit(data)
        clusters = model.predict(data)

        # Create umap
        fig = plt.figure(figsize=(10, 5))

        fit = umap.UMAP()
        input_umap = fit.fit_transform(data)

        plt.scatter(input_umap[:, 0], input_umap[:, 1], c=clusters)
        fig.savefig(f"{self.results_folder}/{name}_k_means_clusters.png")
        plt.close('all')

        return metrics.silhouette_score(data, clusters, metric='euclidean'), metrics.calinski_harabasz_score(data,
                                                                                                             clusters)

    def __create_phenograph_cluster(self, data, name):
        print("creating phenograph clusters ...")

        communities, graph, Q = phenograph.cluster(data)
        clusters = pd.Series(communities)
        clusters.to_csv(f'{self.results_folder}/{name}_pg_clusters.csv', index=False)
        pg_clusters = pd.Series(communities)

        fit = umap.UMAP()
        latent_umap = fit.fit_transform(data)

        plot = sns.scatterplot(data=latent_umap, x=latent_umap[:, 0], y=latent_umap[:, 1], hue=pg_clusters)
        fig = plot.get_figure()
        fig.savefig(f"{self.results_folder}/{name}_pg_clusters.png")
        plt.close('all')

        # means = []
        # for i in range(0, pd.Series(communities).max() + 1):
        # cells_index = pg_clusters[pg_clusters == i].index
        # filtered_markers_df = data[data.index.isin(cells_index)]
        # means.append(filtered_markers_df.mean().values)

        # fig = px.imshow(means, x=data.columns)
        # fig.show()

        return metrics.silhouette_score(data, clusters, metric='euclidean'), metrics.calinski_harabasz_score(data,
                                                                                                             clusters)

    def __resolve_group(self, i: int) -> str:
        if i <= 2:
            return "9_2_1"

        elif i <= 5:
            return "9_2_2"

        elif i <= 8:
            return "9_3_1"

        elif i <= 11:
            return "9_3_2"
