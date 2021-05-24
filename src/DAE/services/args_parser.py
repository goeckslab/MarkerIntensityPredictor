import argparse


class ArgumentParser:

    @staticmethod
    def get_args():
        """
        Load all provided cli args
        """
        parser = argparse.ArgumentParser(description='Arguments for the marker intensity cli')
        parser.add_argument("-f", "--file", type=str, required=True, action="store", help="The file to load and use")
        return parser.parse_args()
