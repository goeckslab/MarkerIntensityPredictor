#!/bin/bash

source venv/bin/activate


# Ludwig w/ hyper
python3 get_scores_per_biopsy.py -b mesmer/tumor_in_patient_hyper/9_2_1
python3 get_scores_per_biopsy.py -b mesmer/tumor_in_patient_hyper/9_2_2
python3 get_scores_per_biopsy.py -b mesmer/tumor_in_patient_hyper/9_3_1
python3 get_scores_per_biopsy.py -b mesmer/tumor_in_patient_hyper/9_3_2
python3 get_scores_per_biopsy.py -b mesmer/tumor_in_patient_hyper/9_14_1
python3 get_scores_per_biopsy.py -b mesmer/tumor_in_patient_hyper/9_14_2
python3 get_scores_per_biopsy.py -b mesmer/tumor_in_patient_hyper/9_15_1
python3 get_scores_per_biopsy.py -b mesmer/tumor_in_patient_hyper/9_15_2

python3 get_scores_per_biopsy.py -b mesmer/tumor_out_patient_hyper/9_2_1
python3 get_scores_per_biopsy.py -b mesmer/tumor_out_patient_hyper/9_2_2
python3 get_scores_per_biopsy.py -b mesmer/tumor_out_patient_hyper/9_3_1
python3 get_scores_per_biopsy.py -b mesmer/tumor_out_patient_hyper/9_3_2
python3 get_scores_per_biopsy.py -b mesmer/tumor_out_patient_hyper/9_14_1
python3 get_scores_per_biopsy.py -b mesmer/tumor_out_patient_hyper/9_14_2
python3 get_scores_per_biopsy.py -b mesmer/tumor_out_patient_hyper/9_15_1
python3 get_scores_per_biopsy.py -b mesmer/tumor_out_patient_hyper/9_15_2

