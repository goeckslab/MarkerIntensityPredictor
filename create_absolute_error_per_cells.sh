
source venv/bin/activate

# No Fe data, no HP
python3 absolute_errors_per_cell/calculate_absolute_error_per_cell_ae.py -t data/tumor_mesmer/9_2_1.csv -p ae/ip/9_2_1/predictions.csv
python3 absolute_errors_per_cell/calculate_absolute_error_per_cell_ae.py -t data/tumor_mesmer/9_2_1.csv -p ae/op/9_2_1/predictions.csv

python3 absolute_errors_per_cell/calculate_absolute_error_per_cell_ae.py -t data/tumor_mesmer/9_2_2.csv -p ae/ip/9_2_2/predictions.csv
python3 absolute_errors_per_cell/calculate_absolute_error_per_cell_ae.py -t data/tumor_mesmer/9_2_2.csv -p ae/op/9_2_2/predictions.csv

python3 absolute_errors_per_cell/calculate_absolute_error_per_cell_ae.py -t data/tumor_mesmer/9_3_1.csv -p ae/ip/9_3_1/predictions.csv
python3 absolute_errors_per_cell/calculate_absolute_error_per_cell_ae.py -t data/tumor_mesmer/9_3_1.csv -p ae/op/9_3_1/predictions.csv

python3 absolute_errors_per_cell/calculate_absolute_error_per_cell_ae.py -t data/tumor_mesmer/9_3_2.csv -p ae/ip/9_3_2/predictions.csv
python3 absolute_errors_per_cell/calculate_absolute_error_per_cell_ae.py -t data/tumor_mesmer/9_3_2.csv -p ae/op/9_3_2/predictions.csv

python3 absolute_errors_per_cell/calculate_absolute_error_per_cell_ae.py -t data/tumor_mesmer/9_14_1.csv -p ae/ip/9_14_1/predictions.csv
python3 absolute_errors_per_cell/calculate_absolute_error_per_cell_ae.py -t data/tumor_mesmer/9_14_1.csv -p ae/op/9_14_1/predictions.csv

python3 absolute_errors_per_cell/calculate_absolute_error_per_cell_ae.py -t data/tumor_mesmer/9_14_2.csv -p ae/ip/9_14_2/predictions.csv
python3 absolute_errors_per_cell/calculate_absolute_error_per_cell_ae.py -t data/tumor_mesmer/9_14_2.csv -p ae/op/9_14_2/predictions.csv

python3 absolute_errors_per_cell/calculate_absolute_error_per_cell_ae.py -t data/tumor_mesmer/9_15_1.csv -p ae/ip/9_15_1/predictions.csv
python3 absolute_errors_per_cell/calculate_absolute_error_per_cell_ae.py -t data/tumor_mesmer/9_15_1.csv -p ae/op/9_15_1/predictions.csv