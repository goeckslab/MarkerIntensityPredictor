import pathlib
import phenograph
import pandas as pd
from pathlib import Path
import umap
import seaborn as sns
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import os
import ntpath


class ClusterAnalysis:
    # The files provided by the arguments
    files = None

    # The directory provided by the arguments
    directory = None
    results_folder = Path("results", "cluster_analysis")
    cluster_assignments = pd.DataFrame(columns=['pca', 'vae', 'non'])
    silhouette_scores = pd.DataFrame(columns=['model', 'score', "file", "cluster"])
    calinski_harabasz_scores = pd.DataFrame(columns=['model', 'score', "file", "cluster"])
    args = None

    def __init__(self, args):
        self.args = args
        self.files = self.args.files
        self.directory = self.args.dir

    def create_mean_score_plots(self):
        if self.directory is not None:
            scores = self.__load_files_in_directory()

        else:
            scores = self.__load_files()

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
            g.fig.suptitle(f"{algo}")

            g.savefig(Path(f"{self.results_folder}/{algo}_mean_scores.png"))
            plt.close('all')

            for cluster in scores["cluster"].unique():
                clusters = data[data["cluster"] == cluster]

                mean_score = clusters.groupby(["file", "model"], as_index=False)["score"].mean()

                mean_score["cluster"] = cluster

                g = sns.catplot(
                    data=mean_score, kind="bar",
                    x="file", y="score", hue="model",
                    ci="sd", palette="dark", alpha=.6, height=6
                )
                g.despine(left=True)
                g.set_axis_labels("", "Cluster scores")
                g.fig.suptitle(f"{algo}")

                g.savefig(Path(f"{self.results_folder}/{algo}_{cluster}_scores.png"))
                plt.close('all')

    def create_cluster(self):
        k: int = 0

        for file in self.files:
            data = pd.read_csv(file)

            # Minimum of 3 columns as pca will have only 3
            if data.shape[1] < 3:
                print("Number of cols is not big enough to represent the expected data.")
                print(f"File: {file}")
                print(f"Shape: {data.shape}")
                input()
                continue

            name = self.args.names[k]
            file = self.__resolve_file(Path(file.name))

            # KMeans clustering
            silhouette, calinksi = self.__create_k_means_cluster(data, name)

            self.silhouette_scores = self.silhouette_scores.append({
                "file": file,
                "model": name,
                "score": silhouette,
                "cluster": "kmeans"
            },
                ignore_index=True)

            self.calinski_harabasz_scores = self.calinski_harabasz_scores.append({
                "file": file,
                "model": name,
                "score": calinksi,
                "cluster": "kmeans"
            },
                ignore_index=True)

            # Phenograph clustering
            silhouette, calinksi = self.__create_phenograph_cluster(data, name)

            self.silhouette_scores = self.silhouette_scores.append({
                "file": file,
                "model": name,
                "score": silhouette,
                "cluster": "pg"
            },
                ignore_index=True)

            self.calinski_harabasz_scores = self.calinski_harabasz_scores.append({
                "file": file,
                "model": name,
                "score": calinksi,
                "cluster": "pg"
            },
                ignore_index=True)

            if k == 2:
                k = 0
            else:
                k += 1

            plt.close('all')

        self.silhouette_scores.to_csv(f"{self.results_folder}/silhouette_scores.csv", index=False)
        self.calinski_harabasz_scores.to_csv(f"{self.results_folder}/calinski_harabasz_scores.csv", index=False)

        g = sns.catplot(
            data=self.silhouette_scores, kind="bar",
            x="cluster", y="score", hue="model",
            ci="sd", palette="dark", alpha=.6, height=6
        )
        g.despine(left=True)
        g.set_axis_labels("", "Cluster scores")
        g.fig.suptitle("Silhouette")

        g.set_xticklabels(rotation=90)
        g.savefig(Path(f"{self.results_folder}/silhouette_scores.png"))
        plt.close('all')

        # data = self.calinski_harabasz_scores[self.calinski_harabasz_scores["cluster"] == cluster]
        g = sns.catplot(
            data=self.calinski_harabasz_scores, kind="bar",
            x="cluster", y="score", hue="model",
            ci="sd", palette="dark", alpha=.6, height=6
        )
        g.despine(left=True)
        g.set_axis_labels("", "Cluster scores")
        g.fig.suptitle("Calinski Harabasz")

        g.set_xticklabels(rotation=90)

        g.savefig(Path(f"{self.results_folder}/calinski_harabasz_scores.png"))
        plt.close('all')

    def __create_k_means_cluster(self, data, name) -> (int, int):
        print("Creating kmeans clusters ...")

        model = KMeans(n_clusters=3, max_iter=100, random_state=1)
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

        plot = sns.scatterplot(data=latent_umap, x=latent_umap[:, 0], y=latent_umap[:, 1], hue=pg_clusters.values)
        fig = plot.get_figure()
        fig.savefig(f"{self.results_folder}/{name}_pg_clusters.png")
        plt.close('all')

        return metrics.silhouette_score(data, clusters, metric='euclidean'), metrics.calinski_harabasz_score(data,
                                                                                                             clusters)

    @staticmethod
    def __resolve_file(file: Path) -> str:
        file = file.with_name(file.name.split('.')[0])
        head, tail = ntpath.split(file)
        return tail or ntpath.basename(head)

    def __load_files(self) -> pd.DataFrame:
        frames = []
        for file in self.files:
            data = pd.read_csv(file)
            if 'silhouette_scores' in file.name:
                data['algo'] = "silhouette"

            elif 'calinski_harabasz_scores' in file.name:
                data['algo'] = "calinski"
            else:
                print("invalid file, skipping")
                continue
            frames.append(data)
        return pd.concat(frames).reset_index()

    def __load_files_in_directory(self) -> pd.DataFrame:
        frames = []
        for (dirpath, dirnames, filenames) in os.walk(self.directory):
            for file in filenames:

                if not file.endswith(".csv"):
                    continue

                path = pathlib.Path(dirpath, file)

                data = pd.read_csv(path)

                if 'silhouette_scores' in file:
                    data['algo'] = "silhouette"

                elif 'calinski_harabasz_scores' in file:
                    data['algo'] = "calinski"
                else:
                    print(f"invalid file {path}, skipping")
                    continue

                frames.append(data)

        scores = pd.concat(frames).reset_index()
        del scores["index"]
        return scores
