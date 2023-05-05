BASE_DIR=$1
BIOPSY=$2
PATIENT=$3
markers=('pRB' 'CD45' 'CK19' 'Ki67' 'aSMA' 'Ecad' 'PR' 'CK14' 'HER2' 'AR' 'CK17' 'p21' 'Vimentin' 'pERK' 'EGFR' 'ER')

for marker in "${markers[@]}"; do
    ./shared_scripts/prepare_config_files.sh $BASE_DIR $BIOPSY $marker $PATIENT
done