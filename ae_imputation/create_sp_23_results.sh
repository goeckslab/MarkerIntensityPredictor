#/bin/bash

source venv/bin/activate

python3 ./ae_imputation/ae.py -m "sp_23" -b 9_2_1 -i 10
python3 ./ae_imputation/ae.py -m "sp_23" -b 9_2_2 -i 10
python3 ./ae_imputation/ae.py -m "sp_23" -b 9_3_1 -i 10
python3 ./ae_imputation/ae.py -m "sp_23" -b 9_3_2 -i 10
python3 ./ae_imputation/ae.py -m "sp_23" -b 9_14_1 -i 10
python3 ./ae_imputation/ae.py -m "sp_23" -b 9_14_2 -i 10
python3 ./ae_imputation/ae.py -m "sp_23" -b 9_15_1 -i 10
python3 ./ae_imputation/ae.py -m "sp_23" -b 9_15_2 -i 10