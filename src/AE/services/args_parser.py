import argparse


class ArgumentParser:

    @staticmethod
    def get_args():
        """
        Load all provided cli args
        """
        parser = argparse.ArgumentParser(description='Arguments for the marker intensity cli')
        parser.add_argument("-f", "--file", type=str, required=False, action="store", help="The file to load and use")
        parser.add_argument("-d", "--dir", type=str, required=False, action="store",
                            help="The directory to use for loading the data")
        return parser.parse_args()
