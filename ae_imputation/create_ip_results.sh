#/bin/bash

source venv/bin/activate

python3 ./ae_imputation/ae.py -m "ip" -b 9_2_1
python3 ./ae_imputation/ae.py -m "ip" -b 9_2_2
python3 ./ae_imputation/ae.py -m "ip" -b 9_3_1
python3 ./ae_imputation/ae.py -m "ip" -b 9_3_2
python3 ./ae_imputation/ae.py -m "ip" -b 9_14_1
python3 ./ae_imputation/ae.py -m "ip" -b 9_14_2
python3 ./ae_imputation/ae.py -m "ip" -b 9_15_1
python3 ./ae_imputation/ae.py -m "ip" -b 9_15_2