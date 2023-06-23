test_id=$1
iterations=$2

if [ "$iterations" == "" ]; then
  iterations=1
fi

echo "Iterations:" ${iterations}

markers=('pRB' 'CD45' 'CK19' 'Ki67' 'aSMA' 'Ecad' 'PR' 'CK14' 'HER2' 'AR' 'CK17' 'p21' 'Vimentin' 'pERK' 'EGFR' 'ER')
#markers=('Ki67')
for i in $(seq 1 $iterations); do
  for marker in "${markers[@]}"; do
    echo "${marker}"
    echo test_id="${test_id}" marker="${marker}"
    make -f makefile run_en test_id="${test_id}" marker="${marker}"
    #make -f makefile plots test_id="${test_id}" marker="${marker}"
  done
done
