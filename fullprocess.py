import ast
import json
import os
import subprocess
import sys

import scoring
import training
import deployment


def _load_config():
    with open("config.json", "r") as f:
        return json.load(f)


def _read_ingested_list(prod_path):
    path = os.path.join(prod_path, "ingestedfiles.txt")
    if not os.path.isfile(path):
        return []
    with open(path, "r") as f:
        raw = f.read().strip()
    try:
        return ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return []


##################Check and read new data
def _has_new_data(config):
    prod_path = config["prod_deployment_path"]
    input_path = config["input_folder_path"]

    previously_ingested = set(_read_ingested_list(prod_path))
    current_csvs = [
        f
        for f in os.listdir(input_path)
        if f.endswith(".csv") and os.path.isfile(os.path.join(input_path, f))
    ]
    return sorted(set(current_csvs) - previously_ingested)


##################Deciding whether to proceed, part 1
##################Checking for model drift
def _read_deployed_score(prod_path):
    with open(os.path.join(prod_path, "latestscore.txt"), "r") as f:
        return float(f.read().strip())


def _model_drift_detected(config):
    """Drift if F1 on latest ingested data (deployed model) is lower than deployed latestscore.txt."""
    prod_path = config["prod_deployment_path"]
    old_score = _read_deployed_score(prod_path)
    # Implemented as scoring.score_deployed_on_ingested_data() in scoring.py (same metric as scoring.py).
    new_score = scoring.score_deployed_on_ingested_data()
    return new_score < old_score, old_score, new_score


##################Deciding whether to proceed, part 2
##################Re-deployment
##################Diagnostics and reporting
def main():
    config = _load_config()

    new_files = _has_new_data(config)
    if not new_files:
        print("No new data to ingest; exiting.")
        sys.exit(0)

    subprocess.run([sys.executable, "ingestion.py"], check=True)

    drift, old_score, new_score = _model_drift_detected(config)
    print(
        f"Drift check: deployed score={old_score}, "
        f"score on latest ingested data with deployed model={new_score}, drift={drift}"
    )

    if not drift:
        print("No model drift; exiting.")
        sys.exit(0)

    training.train_model()
    scoring.score_model()
    deployment.store_model_into_pickle()

    subprocess.run([sys.executable, "reporting.py"], check=True)
    subprocess.run([sys.executable, "apicalls.py"], check=True)


if __name__ == "__main__":
    main()
