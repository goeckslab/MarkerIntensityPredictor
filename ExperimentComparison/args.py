import argparse


class ArgumentParser:

    @staticmethod
    def get_args():
        """
        Load all provided cli args
        """
        parser = argparse.ArgumentParser()
        parser.add_argument("--experiment", "-e", action="store", required=True,
                            help="The name of the experiment which should be evaluated",
                            type=str)
        parser.add_argument("--run", "-r", action="store", required=True,
                            help="The name of the run being run",
                            type=str)

        return parser.parse_args()
