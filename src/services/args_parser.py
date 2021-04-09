import argparse


class ArgumentParser:
    @staticmethod
    def load_args():
        parser = argparse.ArgumentParser(description='Process some integers.')
        parser.add_argument('--dl', default=False, action='store_true')
        parser.add_argument('--file', '-f', type=str, action='store', required=True)
        parser.add_argument('--validation', '-v', type=str, action='store', required=False,
                            help="The validation file which is used to validate the models.")
        parser.add_argument("--multi", "-m", default=False, action="store_true")
        return parser.parse_args()
