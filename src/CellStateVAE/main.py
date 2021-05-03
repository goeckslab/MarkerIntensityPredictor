from entities.cell_state_vae import CellStateVAE
from services.args_parser import ArgumentParser
from entities.ae import AutoEncoder

if __name__ == "__main__":

    args = ArgumentParser.get_args()

    if args.vae:
        print("building vae")
        vae = CellStateVAE()

        vae.load_data()
        vae.split_data()
        vae.build_encoder()
        vae.build_decoder()
        vae.train()
        vae.predict()
        # vae.plot_label_clusters()
        vae.create_plots()
        print("Done")

    else:
        print("builing ae")
        ae = AutoEncoder()
        ae.load_data()
        ae.split_data()
        ae.build_auto_encoder()
        ae.predict()
        ae.plots()

