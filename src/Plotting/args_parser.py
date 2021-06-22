import argparse


class ArgumentParser:

    @staticmethod
    def get_args():
        """
        Load all provided cli args
        """
        parser = argparse.ArgumentParser(description='Arguments for the marker intensity cli')
        parser.add_argument("-r2", "--r2score", type=str, required=False, action="store", help="Generates the r2 score plots")
        parser.add_argument("-f", "--files", type=str, required=True, action="store",
                            help="The directory to use for loading the data")
        return parser.parse_args()
