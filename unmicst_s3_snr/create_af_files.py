import argparse
import numpy as np
import pandas as pd
from pathlib import Path

AF_mapping_9_2_1 = {
    'goat-anti-rabbit': ['pERK', "CCND1", "Ecad", "EGFR", "Ki67", "CK14", "CK17"],
    'A555': ["Vimentin", "ER", "pRB", "CK19", "AR", "CK7"],
    'donkey-anti-mouse': ["Rad51", "PR", "CD45", "p21", "cPARP", "HER2", "aSMA"]
}

AF_mapping_9_2_2 = {
    'goat-anti-rabbit': ['pERK', "CCND1", "Ecad", "EGFR", "Ki67", "CK14", "CK17"],
    'A555': ["Vimentin", "ER", "pRB", "CK19", "AR", "CK7"],
    'donkey-anti-mouse': ["Rad51", "PR", "CD45", "p21", "cPARP", "HER2", "aSMA"]
}

AF_mapping_9_3_1 = {
    'goat-anti-rabbit': ['pERK', "CCND1", "Ecad", "EGFR", "Ki67", "CK14", "CK17"],
    'A555': ["Vimentin", "ER", "pRB", "CK19", "AR", "CK7"],
    'donkey-anti-mouse': ["Rad51", "PR", "CD45", "p21", "cPARP", "HER2", "aSMA"]
}

AF_mapping_9_3_2 = {
    'goat-anti-rabbit': ['pERK', "CCND1", "Ecad", "EGFR", "Ki67", "CK14", "CK17"],
    'A555': ["Vimentin", "ER", "pRB", "CK19", "AR", "CK7"],
    'donkey-anti-mouse': ["Rad51", "PR", "CD45", "p21", "cPARP", "HER2", "aSMA"]
}

AF_mapping_9_14_1 = {
    'A488': ['pERK', 'Ki67', "Ecad", "EGFR", "CD20", "PCNA", "CK14", "LaminA/C", "PR"],
    'A555': ['p53', 'panCK', "SMA", "pRB", "aSMA", "ER", "CK19", "AR", "HLAA"],
    'A647': ['PolICTD', 'CD45', "Vimentin", "p21", "PD1", "HER2", "CK17", "H2ax", "CK5"]
}

AF_mapping_9_14_2 = {
    'A488': ['pERK', 'Ki67', "Ecad", "EGFR", "CD20", "PCNA", "CK14", "LaminA/C", "PR"],
    'A555': ['p53', 'panCK', "SMA", "pRB", "aSMA", "ER", "CK19", "AR", "HLAA"],
    'A647': ['PolICTD', 'CD45', "Vimentin", "p21", "PD1", "HER2", "CK17", "H2ax", "CK5"]
}

AF_mapping_9_15_1 = {
    'A488': ['pERK', 'Ki67', "Ecad", "EGFR", "CD20", "PCNA", "CK14", "LaminA/C", "PR"],
    'A555': ['p53', 'panCK', "SMA", "pRB", "aSMA", "ER", "CK19", "AR", "HLAA"],
    'A647': ['PolICTD', 'CD45', "Vimentin", "p21", "PD1", "HER2", "CK17", "H2ax", "CK5"]
}

AF_mapping_9_15_2 = {
    'A488': ['pERK', 'Ki67', "Ecad", "EGFR", "CD20", "PCNA", "CK14", "LaminA/C", "PR"],
    'A555': ['p53', 'panCK', "SMA", "pRB", "aSMA", "ER", "CK19", "AR", "HLAA"],
    'A647': ['PolICTD', 'CD45', "Vimentin", "p21", "PD1", "HER2", "CK17", "H2ax", "CK5"]
}

mappings = {
    '9_2_1': AF_mapping_9_2_1,
    '9_2_2': AF_mapping_9_2_2,
    '9_3_1': AF_mapping_9_3_1,
    '9_3_2': AF_mapping_9_3_2,
    '9_14_1': AF_mapping_9_14_1,
    '9_14_2': AF_mapping_9_14_2,
    '9_15_1': AF_mapping_9_15_1,
    '9_15_2': AF_mapping_9_15_2
}


def SNR(df, mappings):
    for AF_key in mappings:

        for marker in mappings[AF_key]:
            df[marker] = df[marker] / np.mean(df[AF_key])

    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--biopsy", help="The biopsy to correct", required=True)
    parser.add_argument("-od", "--output_dir", help="Output directory")
    args = parser.parse_args()

    output_dir = args.output_dir
    biopsy = args.biopsy

    biopsy_name = Path(biopsy).stem

    df = pd.read_csv(biopsy)
    mapping = mappings[biopsy_name]

    af_corrected = SNR(df, mapping)
    print(af_corrected)
