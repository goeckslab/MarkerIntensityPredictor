import pandas as pd
import sys
from Plotting.plots import Plots
from pathlib import Path
import logging
import umap


def start(args):
    plots = Path("results", "plots")
    plots.mkdir(parents=True, exist_ok=True)

    if args.r2score:
        logging.info("Creating r2 score plots")
        frames = []

        if args.files is not None and args.legend is not None:
            for i in range(len(args.files)):
                data = pd.read_csv(args.files[i], sep=",")
                if "Model" not in data.columns:
                    data["Model"] = args.legend[i]
                frames.append(data)

        else:
            try:
                linear_data = pd.read_csv(Path("results", "lr", "r2_scores.csv"), sep=",")
                frames.append(linear_data)
            except:
                logging.info("Could not find linear regression r2 scores. Skipping...")

            try:
                ae_data = pd.read_csv(Path("results", "ae", "r2_scores.csv"), sep=",")
                ae_data["Model"] = "AE"
                frames.append(ae_data)
            except:
                logging.info("Could not find auto encoder regression r2 scores. Skipping...")

            try:
                dae_data = pd.read_csv(Path("results", "dae", "r2_scores.csv"), sep=",")
                dae_data["Model"] = "DAE"
                frames.append(dae_data)

            except:
                logging.info("Could not find denoising auto encoder regression r2 scores. Skipping...")

            try:
                dae_data = pd.read_csv(Path("results", "vae", "r2_scores.csv"), sep=",")
                dae_data["Model"] = "VAE"
                frames.append(dae_data)

            except:
                logging.info("Could not find denoising auto encoder regression r2 scores. Skipping...")

        if len(frames) == 0:
            print("No data found. Stopping.")
            sys.exit()

        r2_scores = pd.concat(frames)
        Plots.r2_scores_combined(r2_scores, args.name)

    if args.reconstruction:
        print("Generating reconstructed markers plots.")
        input_data = pd.read_csv(args.files[0], sep=",")
        reconstructed_data = pd.read_csv(args.files[1], sep=",")
        # Create individual heatmap
        Plots.plot_reconstructed_markers(input_data, reconstructed_data, args.name)

    if args.correlation:
        print("Generating correlation heatmaps.")
        frames = []
        for i in range(len(args.files)):
            input_data = pd.read_csv(args.files[i], sep=",")
            input_data.rename(columns={'Unnamed: 0': 'Markers'}, inplace=True)
            # Create individual heatmap

            for model in input_data["Model"].unique():
                data = input_data[input_data["Model"] == model]
                Plots.plot_corr_scatter_plot(data, f"{args.name}")
                Plots.plot_corr_heatmap(data, f"{args.name}", model)

            input_data["File"] = args.name[i]
            frames.append(input_data)

        combined_correlations = pd.concat(frames)
        Plots.plot_combined_corr_plot(combined_correlations)

    if args.cluster:
        print("Generation cluster plots")
        input_data = pd.read_csv(args.files[0], sep=",")
        encoded_data = pd.read_csv(args.files[1], sep=",")

        fit = umap.UMAP()
        input_umap = fit.fit_transform(input_data)

        fit = umap.UMAP()
        latent_umap = fit.fit_transform(encoded_data)
        Plots.latent_space_cluster(input_umap, latent_umap, args.names[0])
