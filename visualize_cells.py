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
    parser.add_argument("--n_channels", action="store", required=False, help="The number of channels", type=int)
    parser.add_argument("--file", "-f", action="store", required=True, help="The zarr file to load")
    parser.add_argument("--iterate_channels", "-ic", action="store_true", required=False, help="The zarr file to load",
                        default=False)
    parser.add_argument("--out", "-o", action="store", required=True, help="The output name of the file")

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    x = zarr.open(args.file, mode="r")

    plt.figure(figsize=(10, 10))
    if not args.iterate_channels:
        for i in range(args.cellcount):
            ax = plt.subplot(int(args.cellcount / 8), 8, i + 1)
            ax.axis("off")
            ax.imshow(x[0, i, ...])
        plt.tight_layout()
        plt.savefig(f"{args.out}.png")

    else:

        rows = int(args.cellcount / 8) if int(args.cellcount / 8) != 0 else 1
        rows = int(rows * args.n_channels / 8)

        for i in range(args.cellcount):
            ax = plt.subplot(rows, 8, i + 1)
            ax.axis("off")
            for channel in range(args.n_channels):
                print(f"Cell {i}: Channel {channel + 1}")
                ax.imshow(x[channel, i, ...])
        plt.tight_layout()
        plt.savefig(f"{args.out}.png")
