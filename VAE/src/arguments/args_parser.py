import argparse


class ArgumentParser:

    @staticmethod
    def get_args():
        """
        Load all provided cli args
        """
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(help='Argument parser for Marker Intensity Predictor', dest="command")
        ArgumentParser.create_model_parser(subparsers)
        ArgumentParser.create_latent_space_exploration_parser(subparsers)

        return parser.parse_args()

    @staticmethod
    def create_model_parser(subparsers):
        """
        The args parser for the auto encoder
        """
        model_parser = subparsers.add_parser("model")
        model_parser.add_argument("--file", action="store", required=True, help="The file used for training the model")
        model_parser.add_argument("--morph", action="store_true", help="Include morphological data", default=False)
        model_parser.add_argument("--remove", action="store_true", default=False,
                                  help="If presents the results folder will not be recreated. "
                                       "This is useful for script executions.")
        model_parser.add_argument("--name", action="store", required=True, help="The name of the experiment being run")

        return

    @staticmethod
    def create_latent_space_exploration_parser(subparsers):
        lspe_parser = subparsers.add_parser("lspe")
        lspe_parser.add_argument("-f", "--file", type=str, required=True, action="store",
                                 help="The file to load and use")
        lspe_parser.add_argument("-m", "--markers", type=str, required=True, action="store",
                                 help="The markers file to load and use")
        lspe_parser.add_argument("-flsp", "--fixed", type=int, required=False, action="store",
                                 help="Which dimension of the latent space should be fixed. "
                                      "If an invalid dimension is given none will be fixed",
                                 default=None)
