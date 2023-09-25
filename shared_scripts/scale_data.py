'''
Prepare data for use in Ludwig.
'''

import pandas as pd
import numpy as np
import argparse
from sklearn.preprocessing import StandardScaler

SHARED_MARKERS = ['pRB', 'CD45', 'CK19', 'Ki67', 'aSMA', 'Ecad', 'PR', 'CK14', 'HER2', 'AR', 'CK17', 'p21', 'Vimentin',
                  'pERK', 'EGFR', 'ER']
NON_MARKERS = ['CellID', 'X_centroid', 'Y_centroid', 'Area', 'MajorAxisLength', 'MinorAxisLength', 'Eccentricity',
               'Solidity', 'Extent', 'Orientation']

if __name__ == "__main__":
    # Argument parsing.
    parser = argparse.ArgumentParser()
    parser.add_argument("x_matrix_file", help="X_matrix file")
    parser.add_argument("output_matrix_file", help="Output matrix file")
    args = parser.parse_args()

    # Read X matrix.
    X_df = pd.read_csv(args.x_matrix_file, delimiter=",", header=0)

    # Set up the final dataframe that will be used for input. Te final dataframe is:
    # - log10(shared markers + 1) and then scaled to [0,1]
    markers_df = X_df[SHARED_MARKERS]
    standard_scaler = StandardScaler()
    markers_df = pd.DataFrame(standard_scaler.fit_transform(np.log10(markers_df + 1)), columns=markers_df.columns)

    markers_df.to_csv(args.output_matrix_file, sep="\t", index=False)
