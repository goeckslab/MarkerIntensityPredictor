from args_parser import ArgumentParser
import pandas as pd
import sys

sys.path.append("..")
from plots import Plots

if __name__ == "__main__":
    args = ArgumentParser.get_args()

    if args.r2score is True:
        print("Creating r2 score plots")
        frames = []

        if args.linear is not None:
            linear_data = pd.read_csv(args.linear[0], sep=",")
            frames.append(linear_data)

        if args.ae is not None:
            ae_data = pd.read_csv(args.ae[0], sep=",")
            ae_data["Model"] = "AE"
            frames.append(ae_data)

        if args.dae is not None:
            dae_data = pd.read_csv(args.dae[0], sep=",")
            dae_data["Model"] = "DAE"
            frames.append(dae_data)

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
        if args.ae is not None:
            input_data = pd.read_csv(args.ae[0], sep=",")
            Plots.plot_corr_heatmap(input_data, "ae")


    else:
        print("No mode selected!")
