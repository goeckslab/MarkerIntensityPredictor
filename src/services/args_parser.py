import argparse


class ArgumentParser:
    @staticmethod
    def load_args():
        parser = argparse.ArgumentParser(description='Process some integers.')
        parser.add_argument('--dl', default=False, action='store_true')
        parser.add_argument('--model', '-m', default="Ridge", action='store', type=str)
        parser.add_argument('--file', '-f', type=str, action='store', required=True)
        return parser.parse_args()
