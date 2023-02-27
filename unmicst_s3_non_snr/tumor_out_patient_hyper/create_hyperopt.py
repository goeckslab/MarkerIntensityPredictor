import argparse
import yaml

if __name__ == '__main__':
    # argparser to parse marker
    parser = argparse.ArgumentParser()
    parser.add_argument("marker", help="The marker")
    parser.add_argument("output_file", help="Output file")
    args = parser.parse_args()

    # create yaml dictionary for the following:
    config = {
        "training": {
            "learning_rate": 0.001,
            "optimizer": {
                "type": "adam"
            }
        },
        "hyperopt": {
            "search_alg": {
                "type": "random",
                "executor": {
                    "type": "ray",
                    "num_samples": 12
                }
            },
            "metric": "mean_absolute_error",
            "split": "validation",
            "output_feature": args.marker,
            "goal": "minimize",
            "parameters": {
                "trainer.optimizer.type": {
                    "space": "choice",
                    "categories": ["sgd", "adam", "adagrad"]
                },
                "trainer.learning_rate": {
                    "type": "float",
                    "space": "loguniform",
                    "lower": 0.0001,
                    "upper": 0.1
                },
                "combiner.num_fc_layers": {
                    "space": "randint",
                    "lower": 1,
                    "upper": 4
                },
                "combiner.type": {
                    "space": "choice",
                    "categories": ["concat", "tabnet"]
                },
            }
        }
    }

    # Write config.
    with open(args.output_file, "w") as output_file:
        yaml.dump(config, output_file)
