import argparse
import os
import ludwig.visualize

def load_args():
    parser = argparse.ArgumentParser(description='Ludwig Visualizer')
    parser.add_argument('--folder', '-f', type=str, action='store', required=True)
    return parser.parse_args()


class LudwigVisualizer:
    folder: str = ""

    def __init__(self, folder: str):
        self.folder = folder

    def create_visualizations(self):
        for subdir, dirs, files in os.walk(self.folder):
            for file in files:
                print(os.path.join(subdir, file))
                ludwig.visualize.learning_curves(
                 train_stats_per_model,
                 output_feature_name,
                 model_names=None,
                 output_directory=None,
                 file_format='pdf'
                )


if __name__ == "__main__":
    args = load_args()
    visualizer = LudwigVisualizer(args.folder)
    visualizer.create_visualizations()

