from AE.ae import AutoEncoder


def start(args):
    ae = AutoEncoder(args)
    ae.load_data()
    ae.build_auto_encoder()
    ae.predict()
    ae.calculate_r2_score()
    ae.create_h5ad_object()
    ae.create_test_predictions()
    ae.create_correlation_data()
    ae.write_created_data_to_disk()
