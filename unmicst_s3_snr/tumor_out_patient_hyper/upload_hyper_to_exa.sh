test_id=$1

markers=('pRB' 'CD45' 'CK19' 'Ki67' 'aSMA' 'Ecad' 'PR' 'CK14' 'HER2' 'AR' 'CK17' 'p21' 'Vimentin' 'pERK' 'EGFR' 'ER')
for marker in "${markers[@]}"; do
  echo "${marker}"
  scp -r ./${test_id}/${marker}/results/hyperopt/  kirchgae@exahead1.ohsu.edu:/home/kirchgae/scratch/unmicst_s3_non_snr/tumor_out_patient_hyper/${test_id}/${marker}/results/hyperopt/
done
