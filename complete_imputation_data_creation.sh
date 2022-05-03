source venv/bin/activate

python3 single_feature_imputation.py -e "EX 9 3 1" --model "EX 9 3 1" "Model" --file "data/HTA9-3_Bx1_HMS_Tumor_quant.csv" --folder "./data" --steps 5 -p "0.2" -r "Single Feature Imputation"
python3 single_feature_imputation.py -e "EX 9 3 1" --model "EX 9 3 1" "Model" --file "data/HTA9-3_Bx1_HMS_Tumor_quant.csv" --folder "./data" --steps 5 -p "0.5" -r "Single Feature Imputation"
python3 single_feature_imputation.py -e "EX 9 3 1" --model "EX 9 3 1" "Model" --file "data/HTA9-3_Bx1_HMS_Tumor_quant.csv" --folder "./data" --steps 5 -p "0.99" -r "Single Feature Imputation"
python3 single_feature_imputation.py -e "EX 9 3 1" --model "EX 9 3 1" "Model" --file "data/HTA9-3_Bx1_HMS_Tumor_quant.csv" --folder "./data" --steps 5 -p "1.0" -r "Single Feature Imputation"

python3 multi_feature_imputation.py -e "EX 9 3 1" --model "EX 9 3 1" "Model" --file "data/HTA9-3_Bx1_HMS_Tumor_quant.csv" --folder "./data" --steps 5 -p "0.2" -r "Multi Feature Imputation"
python3 multi_feature_imputation.py -e "EX 9 3 1" --model "EX 9 3 1" "Model" --file "data/HTA9-3_Bx1_HMS_Tumor_quant.csv" --folder "./data" --steps 5 -p "0.5" -r "Multi Feature Imputation"
python3 multi_feature_imputation.py -e "EX 9 3 1" --model "EX 9 3 1" "Model" --file "data/HTA9-3_Bx1_HMS_Tumor_quant.csv" --folder "./data" --steps 5 -p "0.99" -r "Multi Feature Imputation"
python3 multi_feature_imputation.py -e "EX 9 3 1" --model "EX 9 3 1" "Model" --file "data/HTA9-3_Bx1_HMS_Tumor_quant.csv" --folder "./data" --steps 5 -p "1.0" -r "Multi Feature Imputation"

python3 single_feature_imputation.py -e "EX 9 3 2" --model "EX 9 3 2" "Model" --file "data/HTA9-3_Bx1_HMS_Tumor_quant.csv" --folder "./data" --steps 5 -p "0.2" -r "Single Feature Imputation"
python3 single_feature_imputation.py -e "EX 9 3 2" --model "EX 9 3 2" "Model" --file "data/HTA9-3_Bx1_HMS_Tumor_quant.csv" --folder "./data" --steps 5 -p "0.5" -r "Single Feature Imputation"
python3 single_feature_imputation.py -e "EX 9 3 2" --model "EX 9 3 2" "Model" --file "data/HTA9-3_Bx1_HMS_Tumor_quant.csv" --folder "./data" --steps 5 -p "0.99" -r "Single Feature Imputation"
python3 single_feature_imputation.py -e "EX 9 3 2" --model "EX 9 3 2" "Model" --file "data/HTA9-3_Bx1_HMS_Tumor_quant.csv" --folder "./data" --steps 5 -p "1.0" -r "Single Feature Imputation"

python3 multi_feature_imputation.py -e "EX 9 3 2" --model "EX 9 3 2" "Model" --file "data/HTA9-3_Bx1_HMS_Tumor_quant.csv" --folder "./data" --steps 5 -p "0.2" -r "Multi Feature Imputation"
python3 multi_feature_imputation.py -e "EX 9 3 2" --model "EX 9 3 2" "Model" --file "data/HTA9-3_Bx1_HMS_Tumor_quant.csv" --folder "./data" --steps 5 -p "0.5" -r "Multi Feature Imputation"
python3 multi_feature_imputation.py -e "EX 9 3 2" --model "EX 9 3 2" "Model" --file "data/HTA9-3_Bx1_HMS_Tumor_quant.csv" --folder "./data" --steps 5 -p "0.99" -r "Multi Feature Imputation"
python3 multi_feature_imputation.py -e "EX 9 3 2" --model "EX 9 3 2" "Model" --file "data/HTA9-3_Bx1_HMS_Tumor_quant.csv" --folder "./data" --steps 5 -p "1.0" -r "Multi Feature Imputation"