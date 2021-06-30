from args_parser import ArgumentParser
import Plotting.main as plt
import AE.main as ae
import DAE.main as dae
import LinearRegression.main as lr

if __name__ == "__main__":
    args = ArgumentParser.get_args()
    invoked_parser = args.command

    if invoked_parser == "lr":
        lr.start(args)

    elif invoked_parser == "ae":
        ae.start(args)

    elif invoked_parser == "dae":
        dae.start(args)

    elif invoked_parser == "plt":
        print(args.r2score)
        plt.start(args)
