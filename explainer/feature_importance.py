import sys
from ludwig.explain.gbm import GBMExplainer
from ludwig.explain.explanation import Explanation
from ludwig.api import LudwigModel
import logging
import pandas as pd
from argparse import ArgumentParser
from pathlib import Path
from datetime import datetime
import shap

SHARED_MARKERS = ['pRB', 'CD45', 'CK19', 'Ki67', 'aSMA', 'Ecad', 'PR', 'CK14', 'HER2', 'AR', 'CK17', 'p21', 'Vimentin',
                  'pERK', 'EGFR', 'ER']

logging.root.handlers = []
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("explainer/debug.log"),
                        logging.StreamHandler()
                    ])

save_path = Path("data", "cleaned_data", "feature_importance")

if not save_path.exists():
    save_path.mkdir(parents=True)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--biopsy", "-b", type=str, required=True, help="The biopsy. Format should be 9_2_1")

    args = parser.parse_args()

    biopsy = args.biopsy
    patient = '_'.join(biopsy.split("_")[:2])

    logging.info("Started new run...")
    logging.info("Start time: " + str(datetime.now()))
    logging.info(f"Loaded Biopsy: {biopsy}")
    logging.info(f"Patient: {patient}")

    test_path = Path("data", "cleaned_data", "ground_truth", f"{biopsy}_preprocessed_dataset.tsv")
    logging.debug("Loading test data from path: " + str(test_path))
    try:
        test_data = pd.read_csv(test_path, sep="\t")
    except BaseException as ex:
        logging.error("Could not load data using path: " + str(test_path))
        logging.error(ex)
        sys.exit(1)

    feature_attributions: pd.DataFrame = pd.DataFrame()
    for target in SHARED_MARKERS:
        load_path = Path("mesmer", "tumor_exp_patient", biopsy, target, "results", "experiment_run", "model")
        logging.debug(f"Loading model from path: {load_path}")
        # load the model
        try:
            model: LudwigModel = LudwigModel.load(str(load_path))
        except KeyboardInterrupt as ex:
            logging.debug("Manual abort")
            sys.exit(0)
        except BaseException as ex:
            logging.error("Could not load model using path: " + str(load_path))
            logging.error(ex)
            continue

        try:
            explainer = shap.Explainer(model.predict())
            shap_values = explainer(test_data.drop(columns=[target]).copy())
            print(shap_values)
            input()
        except KeyboardInterrupt as ex:
            logging.debug("Manual abort")
            sys.exit(0)
        except BaseException as ex:
            logging.error("Could not predict shap values")
            logging.error(ex)
            input()

        train_path = Path("data", "tumor_mesmer", "combined", "preprocessed", f"{patient}_excluded_dataset.tsv")
        try:
            train_data = pd.read_csv(train_path, sep="\t")
        except BaseException as ex:
            logging.error("Could not load data using path: " + str(train_path))
            logging.error(ex)
            sys.exit(1)

        # get the explainer values
        try:

            # remove target from test data
            input_df = test_data.drop(columns=[target]).copy()
            explainer = GBMExplainer(model=model, inputs_df=input_df, sample_df=test_data[:200], target=target)
            values = explainer.explain()
            temp_df = {}
            for i in range(len(values.global_explanation.label_explanations[0].feature_attributions)):
                temp_df[values.global_explanation.label_explanations[0].feature_attributions[i].feature_name] = \
                    values.global_explanation.label_explanations[0].feature_attributions[i].attribution

            temp_df["Biopsy"] = biopsy
            temp_df = pd.DataFrame(temp_df, index=[target])

            feature_attributions = pd.concat([feature_attributions, temp_df])

        except KeyboardInterrupt as ex:
            logging.debug("Manual abort")
            sys.exit(0)

        except AttributeError as ex:
            logging.error("Attribute error")
            logging.error(ex)
            sys.exit(0)
        except BaseException as ex:
            logging.error("Could not calculate feature values")
            logging.error(ex)
            sys.exit(0)

    # save the feature attributions
    feature_attributions = pd.DataFrame(feature_attributions)
    cols = feature_attributions.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    feature_attributions = feature_attributions[cols]
    logging.debug(f"Saving feature attributions for biopsy {biopsy}")
    feature_attributions.to_csv(Path(save_path, f"{biopsy}_feature_attributions.csv"))
