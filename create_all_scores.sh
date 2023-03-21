#!/bin/bash

hyper=$1

source venv/bin/activate

echo "Creating scores for Elastic Net"
# EN
python3 get_scores_per_biopsy.py -b unmicst_s3_snr/tumor_in_patient_en/9_2_1
python3 get_scores_per_biopsy.py -b unmicst_s3_snr/tumor_in_patient_en/9_2_2
python3 get_scores_per_biopsy.py -b unmicst_s3_snr/tumor_in_patient_en/9_3_1
python3 get_scores_per_biopsy.py -b unmicst_s3_snr/tumor_in_patient_en/9_3_2
python3 get_scores_per_biopsy.py -b unmicst_s3_snr/tumor_in_patient_en/9_14_1
python3 get_scores_per_biopsy.py -b unmicst_s3_snr/tumor_in_patient_en/9_14_2
python3 get_scores_per_biopsy.py -b unmicst_s3_snr/tumor_in_patient_en/9_15_1
python3 get_scores_per_biopsy.py -b unmicst_s3_snr/tumor_in_patient_en/9_15_2

python3 get_scores_per_biopsy.py -b unmicst_s3_non_snr/tumor_in_patient_en/9_2_1
python3 get_scores_per_biopsy.py -b unmicst_s3_non_snr/tumor_in_patient_en/9_2_2
python3 get_scores_per_biopsy.py -b unmicst_s3_non_snr/tumor_in_patient_en/9_3_1
python3 get_scores_per_biopsy.py -b unmicst_s3_non_snr/tumor_in_patient_en/9_3_2
python3 get_scores_per_biopsy.py -b unmicst_s3_non_snr/tumor_in_patient_en/9_14_1
python3 get_scores_per_biopsy.py -b unmicst_s3_non_snr/tumor_in_patient_en/9_14_2
python3 get_scores_per_biopsy.py -b unmicst_s3_non_snr/tumor_in_patient_en/9_15_1
python3 get_scores_per_biopsy.py -b unmicst_s3_non_snr/tumor_in_patient_en/9_15_2

python3 get_scores_per_biopsy.py -b mesmer/tumor_in_patient_en/9_2_1
python3 get_scores_per_biopsy.py -b mesmer/tumor_in_patient_en/9_2_2
python3 get_scores_per_biopsy.py -b mesmer/tumor_in_patient_en/9_3_1
python3 get_scores_per_biopsy.py -b mesmer/tumor_in_patient_en/9_3_2
python3 get_scores_per_biopsy.py -b mesmer/tumor_in_patient_en/9_14_1
python3 get_scores_per_biopsy.py -b mesmer/tumor_in_patient_en/9_14_2
python3 get_scores_per_biopsy.py -b mesmer/tumor_in_patient_en/9_15_1
python3 get_scores_per_biopsy.py -b mesmer/tumor_in_patient_en/9_15_2

python3 get_scores_per_biopsy.py -b unmicst_s3_snr/tumor_out_patient_en/9_2_1
python3 get_scores_per_biopsy.py -b unmicst_s3_snr/tumor_out_patient_en/9_2_2
python3 get_scores_per_biopsy.py -b unmicst_s3_snr/tumor_out_patient_en/9_3_1
python3 get_scores_per_biopsy.py -b unmicst_s3_snr/tumor_out_patient_en/9_3_2
python3 get_scores_per_biopsy.py -b unmicst_s3_snr/tumor_out_patient_en/9_14_1
python3 get_scores_per_biopsy.py -b unmicst_s3_snr/tumor_out_patient_en/9_14_2
python3 get_scores_per_biopsy.py -b unmicst_s3_snr/tumor_out_patient_en/9_15_1
python3 get_scores_per_biopsy.py -b unmicst_s3_snr/tumor_out_patient_en/9_15_2

python3 get_scores_per_biopsy.py -b unmicst_s3_non_snr/tumor_out_patient_en/9_2_1
python3 get_scores_per_biopsy.py -b unmicst_s3_non_snr/tumor_out_patient_en/9_2_2
python3 get_scores_per_biopsy.py -b unmicst_s3_non_snr/tumor_out_patient_en/9_3_1
python3 get_scores_per_biopsy.py -b unmicst_s3_non_snr/tumor_out_patient_en/9_3_2
python3 get_scores_per_biopsy.py -b unmicst_s3_non_snr/tumor_out_patient_en/9_14_1
python3 get_scores_per_biopsy.py -b unmicst_s3_non_snr/tumor_out_patient_en/9_14_2
python3 get_scores_per_biopsy.py -b unmicst_s3_non_snr/tumor_out_patient_en/9_15_1
python3 get_scores_per_biopsy.py -b unmicst_s3_non_snr/tumor_out_patient_en/9_15_2

python3 get_scores_per_biopsy.py -b mesmer/tumor_out_patient_en/9_2_1
python3 get_scores_per_biopsy.py -b mesmer/tumor_out_patient_en/9_2_2
python3 get_scores_per_biopsy.py -b mesmer/tumor_out_patient_en/9_3_1
python3 get_scores_per_biopsy.py -b mesmer/tumor_out_patient_en/9_3_2
python3 get_scores_per_biopsy.py -b mesmer/tumor_out_patient_en/9_14_1
python3 get_scores_per_biopsy.py -b mesmer/tumor_out_patient_en/9_14_2
python3 get_scores_per_biopsy.py -b mesmer/tumor_out_patient_en/9_15_1
python3 get_scores_per_biopsy.py -b mesmer/tumor_out_patient_en/9_15_2

# Ludwig w/o hyper
echo "Creating scores for Ludwig w/o hyper"

python3 get_scores_per_biopsy.py -b unmicst_s3_snr/tumor_in_patient/9_2_1
python3 get_scores_per_biopsy.py -b unmicst_s3_snr/tumor_in_patient/9_2_2
python3 get_scores_per_biopsy.py -b unmicst_s3_snr/tumor_in_patient/9_3_1
python3 get_scores_per_biopsy.py -b unmicst_s3_snr/tumor_in_patient/9_3_2
python3 get_scores_per_biopsy.py -b unmicst_s3_snr/tumor_in_patient/9_14_1
python3 get_scores_per_biopsy.py -b unmicst_s3_snr/tumor_in_patient/9_14_2
python3 get_scores_per_biopsy.py -b unmicst_s3_snr/tumor_in_patient/9_15_1
python3 get_scores_per_biopsy.py -b unmicst_s3_snr/tumor_in_patient/9_15_2

python3 get_scores_per_biopsy.py -b unmicst_s3_non_snr/tumor_in_patient/9_2_1
python3 get_scores_per_biopsy.py -b unmicst_s3_non_snr/tumor_in_patient/9_2_2
python3 get_scores_per_biopsy.py -b unmicst_s3_non_snr/tumor_in_patient/9_3_1
python3 get_scores_per_biopsy.py -b unmicst_s3_non_snr/tumor_in_patient/9_3_2
python3 get_scores_per_biopsy.py -b unmicst_s3_non_snr/tumor_in_patient/9_14_1
python3 get_scores_per_biopsy.py -b unmicst_s3_non_snr/tumor_in_patient/9_14_2
python3 get_scores_per_biopsy.py -b unmicst_s3_non_snr/tumor_in_patient/9_15_1
python3 get_scores_per_biopsy.py -b unmicst_s3_non_snr/tumor_in_patient/9_15_2

python3 get_scores_per_biopsy.py -b mesmer/tumor_in_patient/9_2_1
python3 get_scores_per_biopsy.py -b mesmer/tumor_in_patient/9_2_2
python3 get_scores_per_biopsy.py -b mesmer/tumor_in_patient/9_3_1
python3 get_scores_per_biopsy.py -b mesmer/tumor_in_patient/9_3_2
python3 get_scores_per_biopsy.py -b mesmer/tumor_in_patient/9_14_1
python3 get_scores_per_biopsy.py -b mesmer/tumor_in_patient/9_14_2
python3 get_scores_per_biopsy.py -b mesmer/tumor_in_patient/9_15_1
python3 get_scores_per_biopsy.py -b mesmer/tumor_in_patient/9_15_2

python3 get_scores_per_biopsy.py -b unmicst_s3_snr/tumor_out_patient/9_2_1
python3 get_scores_per_biopsy.py -b unmicst_s3_snr/tumor_out_patient/9_2_2
python3 get_scores_per_biopsy.py -b unmicst_s3_snr/tumor_out_patient/9_3_1
python3 get_scores_per_biopsy.py -b unmicst_s3_snr/tumor_out_patient/9_3_2
python3 get_scores_per_biopsy.py -b unmicst_s3_snr/tumor_out_patient/9_14_1
python3 get_scores_per_biopsy.py -b unmicst_s3_snr/tumor_out_patient/9_14_2
python3 get_scores_per_biopsy.py -b unmicst_s3_snr/tumor_out_patient/9_15_1
python3 get_scores_per_biopsy.py -b unmicst_s3_snr/tumor_out_patient/9_15_2

python3 get_scores_per_biopsy.py -b unmicst_s3_non_snr/tumor_out_patient/9_2_1
python3 get_scores_per_biopsy.py -b unmicst_s3_non_snr/tumor_out_patient/9_2_2
python3 get_scores_per_biopsy.py -b unmicst_s3_non_snr/tumor_out_patient/9_3_1
python3 get_scores_per_biopsy.py -b unmicst_s3_non_snr/tumor_out_patient/9_3_2
python3 get_scores_per_biopsy.py -b unmicst_s3_non_snr/tumor_out_patient/9_14_1
python3 get_scores_per_biopsy.py -b unmicst_s3_non_snr/tumor_out_patient/9_14_2
python3 get_scores_per_biopsy.py -b unmicst_s3_non_snr/tumor_out_patient/9_15_1
python3 get_scores_per_biopsy.py -b unmicst_s3_non_snr/tumor_out_patient/9_15_2

python3 get_scores_per_biopsy.py -b mesmer/tumor_out_patient/9_2_1
python3 get_scores_per_biopsy.py -b mesmer/tumor_out_patient/9_2_2
python3 get_scores_per_biopsy.py -b mesmer/tumor_out_patient/9_3_1
python3 get_scores_per_biopsy.py -b mesmer/tumor_out_patient/9_3_2
python3 get_scores_per_biopsy.py -b mesmer/tumor_out_patient/9_14_1
python3 get_scores_per_biopsy.py -b mesmer/tumor_out_patient/9_14_2
python3 get_scores_per_biopsy.py -b mesmer/tumor_out_patient/9_15_1
python3 get_scores_per_biopsy.py -b mesmer/tumor_out_patient/9_15_2

if [ -z "$hyper" ]; then
  echo "Hyperopt scores will not be created"
  exit 0
else
  echo "Hyperopt scores will now be created"
fi

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


# Ludwig w/ hyper + fe

echo "Load FE data"
