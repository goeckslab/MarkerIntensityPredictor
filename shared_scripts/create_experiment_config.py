import argparse
import yaml, json
from pathlib import Path

if __name__ == '__main__':
    # argparser to parse marker
    parser = argparse.ArgumentParser()
    parser.add_argument("biopsy", help="The biopsy")
    parser.add_argument("marker", help="The marker")
    parser.add_argument("output_file", help="Output file")
    args = parser.parse_args()

    marker = args.marker
    biopsy = args.biopsy

    path = Path(biopsy, f"{marker}", "results", "hyperopt", "hyperopt_statistics.json")
    f = open(path)
    data = json.load(f)
    # print(data["hyperopt_results"][0]["parameters"])
    # print(data["hyperopt_results"][0]["parameters"]["combiner.num_fc_layers"])
    early_stop: 10
    # create yaml dictionary for the following:
    config = {
        "model_type": "gbm",
        "combiner": {
            "type": data["hyperopt_results"][0]["parameters"]["combiner.type"],
            "num_fc_layers": data["hyperopt_results"][0]["parameters"]["combiner.num_fc_layers"]
        },
        "training": {
            "learning_rate": data["hyperopt_results"][0]["parameters"]["trainer.learning_rate"],
            "optimizer": {
                "type": data["hyperopt_results"][0]["parameters"]["trainer.optimizer.type"]
            },
            "early_stop": 10
        },
    }
    # Write config.
    with open(args.output_file, "w") as output_file:
        yaml.dump(config, output_file)
