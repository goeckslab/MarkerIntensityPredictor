from entities.ae import AutoEncoder

if __name__ == "__main__":
    ae = AutoEncoder()
    ae.load_data()
    ae.split_data()
    ae.build_auto_encoder()
    ae.predict()
    ae.create_h5ad_object()
    ae.plots()
