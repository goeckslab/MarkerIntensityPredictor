import argparse
import pandas as pd
from pathlib import Path

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", "-f", type=Path, required=True, help="The file to convert")
    args = parser.parse_args()

    file: Path = args.file

    df = pd.read_csv(file)
    # convert FE columns according to the following mapping
    mapping = {23: 15, 46: 30, 92: 60, 138: 90, 184: 120}
    df["FE"] = df["FE"].replace(mapping)


    save_path: Path = Path("converted")
    model: str = file.parent.name
    save_path = Path(save_path, model)
    if not save_path.exists():
        save_path.mkdir(parents=True, exist_ok=True)
    df.to_csv(Path(save_path, f"{file.stem}.csv"), index=False)
