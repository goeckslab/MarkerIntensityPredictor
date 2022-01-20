import argparse


class ArgumentsParser:
    @staticmethod
    def get_args():
        """
        Load all provided cli args
        """
        parser = argparse.ArgumentParser()
        parser.add_argument('--file', '-f', type=argparse.FileType('r'), action='store', required=True,
                               help='The input file to process')

        return parser.parse_args()
