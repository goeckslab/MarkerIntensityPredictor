import pandas as pd
import argparse
from dieboldmariano import dm_test
from scipy import stats

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--biopsy", help="biopsy to check results", required=True)
    parser.add_argument("-op", action="store_true", default=False)
    args = parser.parse_args()

    out_patient = args.op
    if out_patient:
        print("Evaluating out_patient biopsy")

    biopsy = args.biopsy

    patient = "in_patient" if not out_patient else "out_patient"

    truth = pd.read_csv(f"data/tumor_mesmer/preprocessed/{biopsy}_preprocessed_dataset.tsv", sep="\t")
    en_errors = pd.read_csv(f"data/absolute_error_per_cell/tumor_{patient}_en/{biopsy}.csv")
    ludwig_errors = pd.read_csv(f"data/absolute_error_per_cell/tumor_{patient}/{biopsy}.csv")
    ludwig_hyper_errors = pd.read_csv(f"data/absolute_error_per_cell/tumor_{patient}_hyper/{biopsy}.csv")
    ludwig_sp_46_errors = pd.read_csv(f"data/absolute_error_per_cell/tumor_{patient}_sp_46/{biopsy}.csv")

    for marker in truth.columns:
        print(f"Marker: {marker}")
        # print variance for each method
        print("EN: ", en_errors[marker].var())
        print("Ludwig: ", ludwig_errors[marker].var())
        print("Ludwig hyper: ", ludwig_hyper_errors[marker].var())
        print("Ludwig 46: ", ludwig_sp_46_errors[marker].var())
        results = stats.ttest_ind(en_errors[marker], ludwig_errors[marker], equal_var=False)
        print(results)
        results = stats.ttest_ind(en_errors[marker], ludwig_hyper_errors[marker], equal_var=False)
        print(results)
        results = stats.ttest_ind(en_errors[marker], ludwig_sp_46_errors[marker], equal_var=False)
        print(results)
        input()
