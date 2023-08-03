'''
Create input features YAML for Ludwig configuration.
'''

import yaml
import argparse
import pandas as pd


if __name__ == "__main__":
    # Argument parsing.
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", help="Input file")
    parser.add_argument("target", help="Feature to predict")
    parser.add_argument("output_file", help="Output file")
    parser.add_argument("--spatial", help="Spatial radius", default=0, type=int)


    args = parser.parse_args()
    target = args.target
    spatial_radius: int = int(args.spatial)

    # Read features from input.
    input_df_cols = pd.read_csv(args.input_file, delimiter="\t", header=0, nrows=0).columns.tolist()
    input_df_cols.remove(target)
    if spatial_radius != 0:
        print(f"Spatial is set to {spatial_radius}")
        input_df_cols.remove(f"{target}_mean")
        assert len(input_df_cols) == 30, f"Wrong number of input features for spatial radius {spatial_radius}"
    else:
        assert len(input_df_cols) == 15, f"Wrong number of input features for spatial radius {spatial_radius}"




    # Define inputs.
    inputs = [{"name": c, "type": "numerical"} for c in input_df_cols]
    outputs = [{"name": target, "type": "numerical"}]
    config = {
        "input_features": inputs,
        "output_features": outputs
    }
    
    # Write config.
    with open(args.output_file, "w") as output_file:
        yaml.dump(config, output_file)    



