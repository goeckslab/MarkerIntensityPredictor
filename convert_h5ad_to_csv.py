import scanpy as sc
import argparse
import pandas as pd

if __name__ == "__main__":
    # Argument parsing.
    parser = argparse.ArgumentParser()
    parser.add_argument("x_matrix_file", help="X_matrix file")
    parser.add_argument("output_matrix_file", help="Output matrix file")
    args = parser.parse_args()

    # load h5ad file and convert to csv
    adata = sc.read_h5ad(args.x_matrix_file)

    # extract x from adata file and convert to df
    df = pd.DataFrame(adata.X, columns=adata.var_names)
    df["X_centroid"] = adata.obs["X_centroid"].values
    df["Y_centroid"] = adata.obs["Y_centroid"].values

    # Convert E-cdaherin to Ecad
    df = df.rename(columns={"E-cadherin": "Ecad"})
    # Convert p-ERK to pERK
    df = df.rename(columns={"p-ERK": "pERK"})
    df = df.rename(columns={"ERK-1": "pERK"})
    # Convert p-Rb to pRB
    df = df.rename(columns={"p-Rb": "pRB"})
    df = df.rename(columns={"Rb": "pRB"})

    df.to_csv(args.output_matrix_file, sep=",", index=False)
