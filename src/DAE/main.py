from entities.dae import DenoisingAutoEncoder
import numpy as np

if __name__ == "__main__":
    data = np.random.rand(800, 4)
    print(data)
    dae = DenoisingAutoEncoder()
    dae.load_data()
    dae.split_data()
    dae.add_noise()
    dae.build_auto_encoder()
    dae.predict()
    dae.calculate_r2_score()
    dae.create_h5ad_object()
    dae.k_means()
