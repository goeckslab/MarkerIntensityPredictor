import pandas as pd
from pathlib import Path
from typing import Dict
# Use this to map the core to the patient

core_patient_map: Dict = {
    "A01": "HTA14_1",
    "A02": "HTA14_2",
    "A03": "HTA14_3",
    "A04": "HTA14_4",
    "A05": "HTA14_5",
    "A06": "HTA14_6",
    "A07": "HTA14_7",
    "A08": "HTA14_8",
    "A09": "HTA14_9",
    "A10": "HTA14_10",
    "A11": "HTA14_11",
    "B01": "HTA14_12",
    "B02": "HTA14_13",
    "B03": "HTA14_14",
    "B04": "HTA14_15",
    "B05": "HTA14_16",
    "B06": "HTA14_17",
    "B07": "HTA14_18",
    "B08": "HTA14_19",
    "B09": "HTA14_12",
    "B10": "HTA14_20",
    "B11": "HTA14_21",
    "C01": "HTA14_22",
    "C02": "HTA14_23",
    "C03": "HTA14_24",
    "C04": "HTA14_25",
    "C05": "HTA14_8",
    "C06": "HTA14_9",
    "C07": "HTA14_26",
    "C08": "HTA14_24",
    "C09": "HTA14_27",
    "C10": "HTA14_28",
    "C11": "HTA14_29",
    "D01": "HTA14_6",
    "D02": "HTA14_30",
    "D03": "HTA14_31",
    "D04": "HTA14_32",
    "D05": "HTA14_33",
    "D06": "HTA14_20",
    "D07": "HTA14_34",
    "D08": "HTA14_35",
    "D09": "HTA14_36",
    "D10": "HTA14_2",
    "D11": "HTA14_37",
    "E01": "HTA14_38",
    "E02": "HTA14_4",
    "E03": "HTA14_39",
    "E04": "HTA14_5",
    "E05": "HTA14_40",
    "E06": "HTA14_3",
    "E07": "HTA14_14",
    "E08": "HTA14_22",
    "E09": "HTA14_32",
    "E10": "HTA14_41",
    "E11": "HTA14_42",
    "F01": "HTA14_27",
    "F02": "HTA14_26",
    "F03": "HTA14_7",
    "F04": "HTA14_43",
    "F05": "HTA14_44",
    "F06": "HTA14_38",
    "F07": "HTA14_30",
    "F08": "HTA14_10",
    "F09": "HTA14_45",
    "F10": "HTA14_31",
    "F11": "HTA14_46",
    "G01": "HTA14_40",
    "G02": "HTA14_34",
    "G03": "HTA14_45",
    "G04": "HTA14_35",
    "G05": "HTA14_1",
    "G06": "HTA14_41",
    "G07": "HTA14_23",
    "G08": "HTA14_39",
    "G09": "HTA14_17",
    "G10": "HTA14_25",
    "G11": "HTA14_47",
    "H01": "HTA14_48",
    "H02": "HTA14_49",
    "H03": "HTA14_18",
    "H04": "HTA14_28",
    "H05": "HTA14_50",
    "H06": "HTA14_16",
    "H07": "HTA14_43",
    "H08": "HTA14_49",
    "H09": "HTA14_15",
    "H10": "HTA14_44",
    "H11": "HTA14_51"
}

SHARED_MARKERS = ['pRB', 'CD45', 'CK19', 'Ki67', 'aSMA', 'Ecad', 'PR', 'CK14', 'HER2', 'AR', 'CK17', 'p21', 'Vimentin',
                  'pERK', 'EGFR', 'ER']

biopsy_folder: Path = Path("data", "cleaned_data", "tma", "biopsies")

if __name__ == '__main__':
    # clean biopsy folder
    for file in biopsy_folder.iterdir():
        if file.is_file():
            file.unlink()

    # load tma data
    tma_data = pd.read_csv(Path("data", "cleaned_data", "tma", "cores", "tma_single_cell.tsv"), sep="\t")
    print(tma_data["Core"].unique())

    for core in tma_data['Core'].unique():
        biopsy = tma_data[tma_data['Core'] == core]
        # remove core and subtype
        biopsy = biopsy.drop(columns=["Core", "Subtype"])

        # select only shared markers
        biopsy = biopsy[SHARED_MARKERS]

        core_number: int = int(core[1:])
        if core_number < 10:
            core = f"{core[0]}0{core_number}"

        patient_id = core_patient_map[core]

        file_number: int = 0
        save_path = Path(biopsy_folder, f"{patient_id}_bx_{file_number}.tsv")
        while save_path.exists():
            file_number += 1
            save_path = Path(biopsy_folder, f"{patient_id}_bx_{file_number}.tsv")

        # save biopsy
        biopsy.to_csv(save_path, sep="\t", index=False)