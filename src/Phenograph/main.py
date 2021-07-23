import phenograph
from args_parser import ArgumentParser
import pandas as pd
from pathlib import Path
import plotly.express as px
import umap
import seaborn as sns


class PhenographCluster:
    results_folder = Path("results")

    def __init__(self):
        args = ArgumentParser.get_args()

        data = pd.read_csv(args.file)
        communities, graph, Q = phenograph.cluster(data)
        print(len(communities))
        print(communities)
        clusters = pd.Series(communities)
        clusters.to_csv(f'{self.results_folder}/pg_clusters.csv')
        pg_clusters = pd.Series(communities)

        fit = umap.UMAP()

        latent_umap = fit.fit_transform(data)

        plot = sns.scatterplot(data=latent_umap, x=latent_umap[:, 0], y=latent_umap[:, 1], hue=pg_clusters)
        fig = plot.get_figure()
        fig.savefig(f"{self.results_folder}/scatter.png")


        means = []
        for i in range(0, pd.Series(communities).max() + 1):
            cells_index = pg_clusters[pg_clusters == i].index
            filtered_markers_df = data[data.index.isin(cells_index)]
            means.append(filtered_markers_df.mean().values)

        fig = px.imshow(means, x=data.columns)
        fig.show()
