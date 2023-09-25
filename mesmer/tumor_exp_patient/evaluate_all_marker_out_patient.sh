test_id=$1
patient=$2

markers=('pRB' 'CD45' 'CK19' 'Ki67' 'aSMA' 'Ecad' 'PR' 'CK14' 'HER2' 'AR' 'CK17' 'p21' 'Vimentin' 'pERK' 'EGFR' 'ER')
#markers=('pERK' 'EGFR' 'ER')

for marker in "${markers[@]}"; do
  echo "${marker}"
  echo test_id="${test_id}" marker="${marker}" patient_id="${patient}"

  make -f makefile ludwig-experiment test_id="${test_id}" marker="${marker}" patient_id="${patient}" random_seed=456
  #make -f makefile ludwig-evaluate test_id="${test_id}" marker="${marker}" patient_id="${patient}" random_seed=456  &&
  #make -f makefile ludwig-plots test_id="${test_id}" marker="${marker}" patient_id="${patient}" random_seed=456  &&
  #make -f makefile clean test_id="${test_id}" marker="${marker}" patient_id="${patient}"  random_seed=456
done
