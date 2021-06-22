import sys
import os
sys.path.append(os.path.split(os.environ['VIRTUAL_ENV'])[0])
from entities.ae import AutoEncoder

if __name__ == "__main__":
    ae = AutoEncoder()
    ae.load_data()
    ae.build_auto_encoder()
    ae.predict()
    ae.calculate_r2_score()
    ae.create_h5ad_object()
    ae.create_val_predictions()
    ae.write_created_data_to_disk()
