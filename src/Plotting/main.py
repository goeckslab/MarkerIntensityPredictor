import pandas as pd
import sys
from Plotting.plots import Plots
from pathlib import Path
import logging


def start(args):
    plots = Path("results", "plots")
    plots.mkdir(parents=True, exist_ok=True)

    if args.r2score is True:
        logging.info("Creating r2 score plots")
        frames = []

        if args.files is not None and args.names is not None:
            for i in range(len(args.files)):
                data = pd.read_csv(args.files[i], sep=",")
                if "Model" not in data.columns:
                    data["Model"] = args.names[i]
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

        if len(frames) == 0:
            print("No data found. Stopping.")
            sys.exit()

        r2_scores = pd.concat(frames)
        Plots.r2_scores_combined(r2_scores)

    if args.reconstructed is True:
        print("Creating reconstructed data plots")
        if args.ae is not None:
            if len(args.ae) != 2:
                print("Need 2 files only!")
                sys.exit()

            input_data = pd.read_csv(args.ae[0], sep=",")
            reconstructed_data = pd.read_csv(args.ae[1], sep=",")

            Plots.plot_reconstructed_markers(input_data, reconstructed_data, "ae")

        if args.dae is not None:
            if len(args.ae) != 2:
                print("Need 2 files only!")
                sys.exit()

            input_data = pd.read_csv(args.ae[0], sep=",")
            reconstructed_data = pd.read_csv(args.ae[1], sep=",")

            Plots.plot_reconstructed_markers(input_data, reconstructed_data, "dae")

    if args.corr is True:
        print("Generating correlation heatmaps.")
        frames = []
        for i in range(len(args.files)):
            input_data = pd.read_csv(args.files[i], sep=",")
            # Create individual heatmap
            Plots.plot_corr_heatmap(input_data, f"ae_{args.names[i]}")

            input_data["File"] = args.names[i]
            frames.append(input_data)

        combined_correlations = pd.concat(frames)
        Plots.plot_combined_corr_plot(combined_correlations)
