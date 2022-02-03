import argparse


class ArgumentParser:

    @staticmethod
    def get_args():
        """
        Load all provided cli args
        """
        parser = argparse.ArgumentParser()
        parser.add_argument("--name", action="store", required=True, help="The name of the experiment being run")
        parser.add_argument("--file", action="store", required=True, help="The file used for training the model")
        parser.add_argument("--morph", action="store_true", help="Include morphological data", default=False)
        return parser.parse_args()
