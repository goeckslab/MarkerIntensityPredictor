import argparse


class ArgumentParser:

    @staticmethod
    def get_args():
        """
        Load all provided cli args
        """
        parser = argparse.ArgumentParser()
        parser.add_argument("--experiment", "-exp", action="store", required=True,
                            help="The name of the experiment being run",
                            type=str)
        parser.add_argument("--group", "-g", action="store", required=False,
                            help="Adds a tag to the experiment indicating to which group it belongs too.",
                            type=str)
        parser.add_argument("--file", action="store", required=True, help="The file used for training the model")
        parser.add_argument("--morph", action="store_true", help="Include morphological data", default=True)
        parser.add_argument("--mode", action="store",
                            help="If used only the given model will be executed and no comparison will take place",
                            required=False, choices=['vae', 'ae', 'none'], default="none")

        return parser.parse_args()
