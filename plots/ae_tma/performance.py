import pandas as pd
import seaborn as sns
from pathlib import Path
import matplotlib.pyplot as plt

save_path = Path("plots", "figures", "supplements", "ae_tma")
if not save_path.exists():
    save_path.mkdir(parents=True, exist_ok=True)

if __name__ == '__main__':
    # load ae tma scores from cleaned data scores folder
    ae_tma_scores = pd.read_csv(Path("data", "cleaned_data", "scores", "ae_tma", "scores.csv"))

    fig = plt.figure(figsize=(10, 5), dpi=300)
    # plot boxen plot, with hue as mode and x as marker and y as mae
    ax = sns.boxenplot(data=ae_tma_scores, x="Marker", y="MAE", hue="Mode", palette="Set2")
    # rotate x axis laels
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(Path(save_path, "performance.png"), dpi=300)
