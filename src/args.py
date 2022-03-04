import argparse


class ArgumentParser:

    @staticmethod
    def get_args():
        """
        Load all provided cli args
        """
        parser = argparse.ArgumentParser()
        parser.add_argument("--run", "-r", action="store", required=True,
                            help="The name of the run being run",
                            type=str)
        parser.add_argument("--tracking_url", "-t", action="store", required=False,
                            help="The tracking url for the mlflow tracking server", type=str)
        parser.add_argument("--experiment", "-e", action="store", required=False,
                            help="Assigns the run to a particular experiment. "
                                 "If the experiment does not exists it will create a new one.",
                            type=str)
        parser.add_argument("--description", "-d", action="store", required=False,
                            help="A description for the experiment to give a broad overview. "
                                 "This is only used when a new experiment is being created. Ignored if experiment exists",
                            type=str)
        parser.add_argument("--file", action="store", required=True, help="The file used for training the model")
        parser.add_argument("--morph", action="store_true", help="Include morphological data", default=True)
        parser.add_argument("--mode", action="store",
                            help="If used only the given model will be executed and no comparison will take place",
                            required=False, choices=['vae', 'ae', 'none'], default="none")

        return parser.parse_args()
