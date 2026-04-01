from flask import Flask, jsonify, request
import pandas as pd
import json
import os
import numpy as np

from diagnostics import (
    model_predictions,
    dataframe_summary,
    execution_time,
    missing_data,
    outdated_packages_list,
)
from scoring import score_model


######################Set up variables for use in our script
app = Flask(__name__)
app.secret_key = "1652d576-484a-49fd-913a-6879acfa6ba4"

with open("config.json", "r") as f:
    config = json.load(f)


def _to_python_types(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, list):
        return [_to_python_types(item) for item in obj]
    if isinstance(obj, dict):
        return {key: _to_python_types(value) for key, value in obj.items()}
    return obj


#######################Prediction Endpoint
@app.route("/prediction", methods=["POST", "OPTIONS"])
def predict():
    # expects JSON: {"dataset_path": "testdata/testdata.csv"}
    data = request.get_json()
    dataset_path = data.get("dataset_path")
    df = pd.read_csv(dataset_path)
    preds = _to_python_types(model_predictions(df))
    return jsonify(preds), 200


#######################Scoring Endpoint
@app.route("/scoring", methods=["GET", "OPTIONS"])
def scoring():
    # check the score of the deployed model
    f1 = _to_python_types(score_model())
    return jsonify({"f1_score": f1}), 200


#######################Summary Statistics Endpoint
@app.route("/summarystats", methods=["GET", "OPTIONS"])
def summarystats():
    # check means, medians, and stds for each column
    stats = _to_python_types(dataframe_summary())
    return jsonify(stats), 200


#######################Diagnostics Endpoint
@app.route("/diagnostics", methods=["GET", "OPTIONS"])
def diagnostics():
    # check timing, percent NA values, and dependency versions
    timings = _to_python_types(execution_time())
    missing = _to_python_types(missing_data())
    packages = _to_python_types(outdated_packages_list())
    return jsonify(
        {
            "execution_time": timings,
            "missing_data": missing,
            "outdated_packages": packages,
        }
    ), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True, threaded=True)
