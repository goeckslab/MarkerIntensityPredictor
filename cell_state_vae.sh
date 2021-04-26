source ./venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

python3 src/CellStateVAE/cell_state_vae.py
