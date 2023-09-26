import argparse
import pandas as pd
from pathlib import Path
from scipy.stats import ks_2samp

rounds = {"9_2": {
    "2": ["pERK"],
    "3": ["Vimentin", "aSMA"],
    "4": ["Ecad", "ER", "PR"],
    "5": ["EGFR", "pRB", "CD45"],
    "6": ["Ki67", "CK19", "p21"],
    "7": ["CK14", "AR"],
    "8": ["CK17", "HER2"],
},
    "9_3": {
        "2": ["pERK"],
        "3": ["Vimentin", "aSMA"],
        "4": ["Ecad", "ER", "PR"],
        "5": ["EGFR", "pRB", "CD45"],
        "6": ["Ki67", "CK19", "p21"],
        "7": ["CK14", "AR"],
        "8": ["CK17", "HER2"],
    },
    "9_14": {
        "2": ["pERK"],
        "3": ["Ki67", "CD45"],
        "4": ["Ecad", "aSMA", "Vimentin"],
        "5": ["pRB", "EGFR", "p21"],
        "7": ["ER", "HER2"],
        "8": ["CK14", "CK19", "CK17"],
        "9": ["AR"],
        "10": ["PR"]
    },
    "9_15": {
        "2": ["pERK"],
        "3": ["Ki67", "CD45"],
        "4": ["Ecad", "aSMA", "Vimentin"],
        "5": ["pRB", "EGFR", "p21"],
        "7": ["ER", "HER2"],
        "8": ["CK14", "CK19", "CK17"],
        "9": ["AR"],
        "10": ["PR"]
    }
}

results_path = Path("data/absolute_error_per_cell")

if __name__ == '__main__':
    # argsparser
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--biopsy", required=True, help="Input file")
    args = parser.parse_args()

    # read in data
    biopsy = pd.read_csv(args.biopsy)
    case_name = "_".join(Path(args.biopsy).stem.split("_")[0:2])
    selected_rounds = rounds[case_name]

    round_list = sorted(selected_rounds.keys())

    results = []
    for i, t_round in enumerate(round_list):
        if i == 0:
            print("Skipping first round")
            continue

        current_markers = selected_rounds[t_round]
        # calculate ks test for each marker of the previous round with the current round
        for marker in current_markers:
            previous_markers = selected_rounds[round_list[i - 1]]
            # calculate ks test for each marker of the previous round with the current round
            for previous_marker in previous_markers:
                # print(f"{t_round} {marker} {previous_marker}")
                stats = ks_2samp(biopsy[previous_marker], biopsy[marker])

                results.append({
                    "round": t_round,
                    "marker": marker,
                    "previous_marker": previous_marker,
                    "p_value": stats.pvalue,
                })

    results = pd.DataFrame(results)
    results_path = f"{Path(results_path)}/{str(args.biopsy).split('/')[-2]}/{Path(args.biopsy).stem}_p_values.csv"
    results.to_csv(results.to_csv(str(results_path), index=False))
