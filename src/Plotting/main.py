import pandas as pd
import sys
from Plotting.plots import Plots
from pathlib import Path
import logging


def start(args):
    if args.r2score is True:
        logging.info("Creating r2 score plots")
        frames = []

        try:
            linear_data = pd.read_csv(Path("results/lr/r2scores.csv"), sep=",")
            frames.append(linear_data)
        except:
            logging.info("Could not find linear regression r2 scores. Skipping...")

        try:
            ae_data = pd.read_csv(Path("results/ae/r2scores.csv"), sep=",")
            ae_data["Model"] = "AE"
            frames.append(ae_data)
        except:
            logging.info("Could not find auto encoder regression r2 scores. Skipping...")

        try:
            dae_data = pd.read_csv(Path("results/ae/r2scores.csv"), sep=",")
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
        if args.ae is not None:
            input_data = pd.read_csv(args.ae[0], sep=",")
            Plots.plot_corr_heatmap(input_data, "ae")
