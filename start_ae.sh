source ./venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

cd src/AE/
python3 main.py $1 $2 $3 $4 $5 $6 $7
