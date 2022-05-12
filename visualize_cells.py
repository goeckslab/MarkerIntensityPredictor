import argparse
import zarr
import matplotlib.pyplot as plt


def get_args():
    """
       Load all provided cli args
       """
    parser = argparse.ArgumentParser()
    parser.add_argument("-cc", "--cellcount", action="store", required=False, type=int,
                        help="How many cells should be visualized", default=64)
    parser.add_argument("-ch", action="store", required=False, help="A specific channel")
    parser.add_argument("--file", "-f", action="store", required=False, help="The zarr file to load")

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    x = zarr.open(args.file, mode="r")

    plt.figure(figsize=(10, 10))
    for i in range(args.cellcount):
        ax = plt.subplot(int(args.cellcount / 8), 8, i + 1)
        ax.axis("off")
        ax.imshow(x[0, i, ...])
    plt.tight_layout()
    plt.savefig("cell_overview.png")
