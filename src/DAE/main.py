from entities.dae import DenoisingAutoEncoder
import numpy as np

if __name__ == "__main__":
    dae = DenoisingAutoEncoder()
    dae.load_data()
    dae.add_noise()
    dae.build_auto_encoder()
    dae.predict()
    dae.calculate_r2_score()
    dae.create_h5ad_object()
    dae.k_means()
    dae.create_val_predictions()
    dae.write_created_data_to_disk()
