import matplotlib.pyplot as plt
import umap
from entities.vae import VAE
import seaborn as sns
from scipy import stats
from matplotlib.pyplot import figure
from pathlib import Path
import logging


class Plots:
    @staticmethod
    def plot_dists(ax, data, title, plot_type=""):
        logging.info("Plotting distributions")
        # Flatten data, `ravel` yields a 1D "view",
        # which is more efficient than creating a 1D copy.
        f_data = data.ravel()

        if plot_type == "density":
            density = stats.gaussian_kde(f_data)
            n, x, _ = plt.hist(f_data, bins=25, histtype="step", density=True)
            ax.plot(x, density(x))
        elif plot_type == "both":
            density = stats.gaussian_kde(f_data)
            n, x, _ = ax.hist(f_data, bins=25, histtype="bar", density=True)
            ax.plot(x, density(x))
        else:
            ax.hist(f_data, bins=32, histtype="bar")
        ax.set_title(title)
        # ax.set_yscale('log')

    @staticmethod
    def plot_model_performance(history, file_name: str):
        logging.info("Plotting model performance")
        figure(num=None, figsize=(6, 4), dpi=90)
        for key in history.history:
            plt.plot(history.history[key], label=key)
        plt.xlabel("Epoch")
        plt.ylabel("Value")
        plt.legend()
        plt.savefig(Path("results", "ae", f"{file_name}.png"))

    @staticmethod
    def latent_space_cluster(input_umap, latent_umap, file_name: str):
        logging.info("Plotting latent space clusters")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), dpi=150)
        plt.subplots_adjust(wspace=0.2)

        ax1.scatter(x=-input_umap[:, 0], y=-input_umap[:, 1])
        ax1.set_title("UMAP Embedding/Projection of Input")
        ax1.set_xlabel("umap1")
        ax1.set_ylabel("umap2")

        ax2.scatter(x=-latent_umap[:, 0], y=-latent_umap[:, 1])
        ax2.set_title("UMAP Embedding/Projection of Latent Space")
        ax2.set_xlabel("umap1")
        ax2.set_ylabel("umap2")

        plt.savefig(Path("results", "ae", f"{file_name}.png"))

    @staticmethod
    def plot_reconstructed_markers(z_grid, x_pred_grid, markers, file_name: str):
        logging.info("Plotting reconstructed markers")
        sns.set_theme()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 9), dpi=300, gridspec_kw={'width_ratios': [1, 5]})
        plt.subplots_adjust(wspace=0.03, hspace=0.1)

        sns.heatmap(z_grid, ax=ax1, cmap="YlGnBu", cbar_kws=dict(use_gridspec=False, location="top"))
        sns.heatmap(x_pred_grid, ax=ax2, xticklabels=markers, yticklabels=False,
                    cbar_kws=dict(use_gridspec=False, location="top", shrink=.25))

        ax1.set_title("Latent Space")
        ax1.set_ylabel("Combinations")

        ax2.set_title("Reconstructed Marker Intensities")
        ax2.set_xlabel("Marker")
        plt.savefig(Path("results", "ae", f"{file_name}.png"))

    @staticmethod
    def plot_markers(X_train, X_test, X_val, markers, file_name: str):
        logging.info("Plotting markers")
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(9, 10), dpi=300, sharex=True)
        sns.heatmap(X_train, ax=ax1, xticklabels=markers)
        sns.heatmap(X_test, ax=ax2, xticklabels=markers)
        sns.heatmap(X_val, ax=ax3, xticklabels=markers)

        ax1.set_title("X Train")
        ax2.set_title("X Test")
        ax3.set_title("X Validation")
        fig.tight_layout()
        plt.savefig(Path("results", "ae", f"{file_name}.png"))

    @staticmethod
    def plot_reconstructed_intensities(vae: any, X_val, markers, file_name: str):
        logging.info("Plotting reconstructed intensities")
        recon_val = vae.predict(X_val)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 10), dpi=300, sharex=True)
        sns.heatmap(X_val, ax=ax1, xticklabels=markers)

        sns.heatmap(recon_val, ax=ax2, xticklabels=markers)

        ax1.set_title("X Validation")
        ax2.set_title("Reconstructed X Validation")
        fig.tight_layout()
        plt.savefig(Path("results", "ae", f"{file_name}.png"))

    @staticmethod
    def plot_distribution_of_latent_variables(encoder, X_train, latent_dim, step_size, z, file_name: str):
        logging.info("Plotting distribution of latent variables")
        # Distribution of latent variables
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), dpi=300)
        plt.subplots_adjust(wspace=0.2, hspace=0.1)

        mean, log_var, latent_variables = encoder.predict(X_train)
        sns.violinplot(data=latent_variables, ax=ax1)

        ## Two visualization options:
        ## Option 1: show the distribution of z_grid
        ## (a matrix of shape (4096, 6) depending on the variables set):
        # sns.violinplot(data=z_grid, ax=ax2)
        #
        ## Option 2: show the latent variables as in z (not the meshgrid,
        ## which is matrix of shape (6, 4) depending on the variables set):
        for i in range(latent_dim):
            x = [i] * step_size
            y = z[i, :]
            ax2.scatter(x, y)
        ax1.set_title("Distribution of latent variables when encoder is fed train data")
        ax1.set_xlabel("Latent Variables")
        ax1.set_ylabel("Values")

        ax2.set_title("Latent variables to explor")
        ax2.set_xlabel("Latent Variables")
        ax2.set_ylabel("Values")

        plt.savefig(Path("results", "ae", f"{file_name}.png"))
