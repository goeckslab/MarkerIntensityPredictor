import argparse


class ArgumentParser:

    @staticmethod
    def get_args():
        """
        Load all provided cli args
        """
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(help='Argument parser for Marker Intensity Predictor', dest="command")

        parser.add_argument("--remove", action="store_true", default=False,
                            help="If presents the results folder will not be recreated. "
                                 "This is useful for script executions.")

        ArgumentParser.create_lr_parser(subparsers)
        ArgumentParser.create_ae_parser(subparsers)
        ArgumentParser.create_plotting_parser(subparsers)
        ArgumentParser.create_vae_parser(subparsers)
        ArgumentParser.create_cluster_parser(subparsers)
        ArgumentParser.create_pca_parser(subparsers)

        return parser.parse_args()

    @staticmethod
    def create_ae_parser(subparsers):
        ae_parser = subparsers.add_parser("ae")
        ae_parser.add_argument("-f", "--file", type=str, required=False, action="store",
                               help="The file to load and use")
        ae_parser.add_argument("-d", "--dir", type=str, required=False, action="store",
                               help="The directory to use for loading the data")

        return

    @staticmethod
    def create_vae_parser(subparsers):
        ae_parser = subparsers.add_parser("vae")
        ae_parser.add_argument("-f", "--file", type=str, required=False, action="store",
                               help="The file to load and use")
        ae_parser.add_argument("-d", "--dir", type=str, required=False, action="store",
                               help="The directory to use for loading the data")

        return

    @staticmethod
    def create_lr_parser(subparsers):
        lr_parser = subparsers.add_parser("lr")
        lr_parser.add_argument('--file', '-f', type=str, action='store', required=True)
        lr_parser.add_argument('--validation', '-v', type=str, action='store', required=False, default=None,
                               help="The validation file which is used to validate the models.")
        lr_parser.add_argument("--multi", "-m", default=False, action="store_true")

    @staticmethod
    def create_dae_parser(subparsers):
        dae_parser = subparsers.add_parser("dae")
        dae_parser.add_argument("-f", "--file", type=str, required=False, action="store",
                                help="The file to load and use")
        dae_parser.add_argument("-d", "--dir", type=str, required=False, action="store",
                                help="The directory to use for loading the data")

        return

    @staticmethod
    def create_pca_parser(subparsers):
        pca_parser = subparsers.add_parser("pca")
        pca_parser.add_argument("-f", "--file", type=str, required=False, action="store",
                                help="The file to load and use")
        pca_parser.add_argument("-d", "--dir", type=str, required=False, action="store",
                                help="The directory to use for loading the data")

    @staticmethod
    def create_cluster_parser(subparsers):
        cluster_parser = subparsers.add_parser("cl")
        cluster_parser.add_argument("-f", "--files", type=argparse.FileType('r'), required=False, action="store",
                                    help="The files used to generate the clusters", nargs='+')
        cluster_parser.add_argument("-n", "--names", type=str, required=False, action="store",
                                    help="The file name", nargs='+')

        return

    @staticmethod
    def create_plotting_parser(subparsers):
        plotting_parser = subparsers.add_parser("plt")

        plotting_parser.add_argument("-f", "--files", type=argparse.FileType('r'), required=False, action="store",
                                     help="The files used to generate the plot", nargs='+')
        plotting_parser.add_argument("-l", "--legend", type=str, required=False, action="store",
                                     help="The data for the legend", nargs='+')
        plotting_parser.add_argument("-n", "--name", type=str, required=False, action="store",
                                     help="The file name")

        # Modes

        plotting_parser.add_argument("-r2", "--r2score", action="store_true", help="Generates the r2 score plots")
        plotting_parser.add_argument("-r", "--reconstructed", action="store_true",
                                     help="Generates the ae reconstructing plots")
        plotting_parser.add_argument("-corr", "--corr", action="store_true", help="Generates the correlation heatmap")
        plotting_parser.add_argument("-cluster", "--cluster", action="store_true", help="Generates a umap cluster plot")
