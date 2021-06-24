import argparse


class ArgumentParser:

    @staticmethod
    def get_args():
        """
        Load all provided cli args
        """
        parser = argparse.ArgumentParser(description='Arguments for the marker intensity cli')
        parser.add_argument("-l", "--linear", type=argparse.FileType('r'), required=False, action="store",
                            help="The data for the linear files", nargs='+')
        parser.add_argument("-ae", "--ae", type=argparse.FileType('r'), required=False, action="store",
                            help="The data for the linear files", nargs='+')
        parser.add_argument("-dae", "--dae", type=argparse.FileType('r'), required=False, action="store",
                            help="The data for the linear files", nargs='+')
        parser.add_argument("-vae", "--vae", type=argparse.FileType('r'), required=False, action="store",
                            help="The data for the linear files", nargs='+')

        # Modes

        parser.add_argument("-r2", "--r2score", action="store_true", help="Generates the r2 score plots")
        parser.add_argument("-r", "--reconstructed", action="store_true", help="Generates the ae reconstructing plots")
        parser.add_argument("-corr", "--corr", action="store_true", help="Generates the correlation heatmap")

        return parser.parse_args()
