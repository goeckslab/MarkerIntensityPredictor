from entities.dae import DenoisingAutoEncoder

if __name__ == "__main__":
    dae = DenoisingAutoEncoder()
    dae.load_data()
    dae.split_data()
    dae.add_noise()
    dae.build_auto_encoder()
    dae.predict()
    dae.create_h5ad_object()
    dae.plots()

