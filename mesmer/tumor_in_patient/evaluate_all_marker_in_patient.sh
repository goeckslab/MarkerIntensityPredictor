sample_id=$1
test_id=$2

markers=('pRB' 'CD45' 'CK19' 'Ki67' 'aSMA' 'Ecad' 'PR' 'CK14' 'HER2' 'AR' 'CK17' 'p21' 'Vimentin' 'pERK' 'EGFR' 'ER')
for marker in "${markers[@]}"; do
  echo "${marker}"
  echo sample_id="${sample_id}" test_id="${test_id}" marker="${marker}"

  make -f makefile ludwig-experiment sample_id="${sample_id}" test_id="${test_id}" marker="${marker}" random_seed=456
  # make -f makefile ludwig-evaluate sample_id="${sample_id}" test_id="${test_id}" marker="${marker}" random_seed=456 && \
  # make -f makefile ludwig-plots sample_id="${sample_id}" test_id="${test_id}" marker="${marker}" random_seed=456 && \
  # make -f makefile clean sample_id="${sample_id}" test_id="${test_id}" marker="${marker}" random_seed=456
done
