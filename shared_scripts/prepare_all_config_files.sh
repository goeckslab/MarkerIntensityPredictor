BASE_DIR=$1
BIOPSY=$2
PATIENT=$3
SPATIAL=$4
markers=('pRB' 'CD45' 'CK19' 'Ki67' 'aSMA' 'Ecad' 'PR' 'CK14' 'HER2' 'AR' 'CK17' 'p21' 'Vimentin' 'pERK' 'EGFR' 'ER')

echo "spatial is set" ${SPATIAL}
for marker in "${markers[@]}"; do
      if [ "$SPATIAL" != "" ]; then
        echo "Preparing ${marker} for ${BIOPSY} with spatial ${SPATIAL}"
        ./shared_scripts/prepare_config_files_sp.sh $BASE_DIR $BIOPSY $marker $PATIENT $SPATIAL
      else
         ./shared_scripts/prepare_config_files.sh $BASE_DIR $BIOPSY $marker $PATIENT
      fi
done