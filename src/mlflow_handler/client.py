import mlflow


def get_mlflow_client(args):
    if args.tracking_url is None:
        return mlflow.tracking.MlflowClient()
    else:
        return mlflow.tracking.MlflowClient(tracking_uri=args.tracking_url)

