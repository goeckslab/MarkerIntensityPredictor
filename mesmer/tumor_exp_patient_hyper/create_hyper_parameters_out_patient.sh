test_id=$1

markers=('pRB' 'CD45' 'CK19' 'Ki67' 'aSMA' 'Ecad' 'PR' 'CK14' 'HER2' 'AR' 'CK17' 'p21' 'Vimentin' 'pERK' 'EGFR' 'ER')
#markers=('pRB')
for marker in "${markers[@]}"; do
  echo "${marker}"
  echo  test_id="${test_id}" marker="${marker}"
  make -f makefile ludwig-hyperopt  test_id="${test_id}" marker="${marker}"
  make -f makefile clean-hyper  test_id="${test_id}" marker="${marker}"
done
