import pandas as pd
import numpy as np
import timeit
import os
import json
import pickle
import subprocess
import sys

##################Load config.json and get environment variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path']) 
model_path = os.path.join(config["output_model_path"])
prod_deployment_path = os.path.join(config["prod_deployment_path"])

##################Function to get model predictions
def model_predictions(test_data):
    #read the deployed model and a test dataset, calculate predictions
    with open(os.path.join(prod_deployment_path, "trainedmodel.pkl"), "rb") as f:
        deployed_model = pickle.load(f)

    X = test_data[["lastmonth_activity", "lastyear_activity", "number_of_employees"]]
    predictions = deployed_model.predict(X)

    return list(predictions)  # return value should be a list containing all predictions

##################Function to get summary statistics
def dataframe_summary():
    # mean, median, and mode for each numeric column (rubric)
    data = pd.read_csv(os.path.join(dataset_csv_path, "finaldata.csv"))
    numeric_data = data[["lastmonth_activity", "lastyear_activity", "number_of_employees"]]

    means = list(numeric_data.mean())
    medians = list(numeric_data.median())
    mode_df = numeric_data.mode()
    if len(mode_df) == 0:
        modes = [float("nan")] * numeric_data.shape[1]
    else:
        modes = [float(mode_df.iloc[0][col]) for col in numeric_data.columns]

    return [means, medians, modes]


##################Function to get missing data percentages
def missing_data():
    # percent NA per numeric column (rubric)
    data = pd.read_csv(os.path.join(dataset_csv_path, "finaldata.csv"))
    numeric_data = data.select_dtypes(include=[np.number])
    n = len(numeric_data)
    if n == 0:
        return []
    missing_percentages = list(numeric_data.isna().sum() / n)
    return missing_percentages

##################Function to get timings
def execution_time():
    #calculate timing of training.py and ingestion.py
    start_ingestion = timeit.default_timer()
    subprocess.run([sys.executable, "ingestion.py"], check=True)
    end_ingestion = timeit.default_timer()

    start_training = timeit.default_timer()
    subprocess.run([sys.executable, "training.py"], check=True)
    end_training = timeit.default_timer()

    return [end_ingestion - start_ingestion, end_training - start_training]  # return a list of 2 timing values in seconds

##################Function to check dependencies
def outdated_packages_list():
    #get a list of packages from requirements.txt
    requirements = []
    with open("requirements.txt", "r") as f:
        for line in f:
            package_line = line.strip()
            if package_line:
                requirements.append(package_line.split("==")[0])

    # pip outdated contains installed/current and latest versions
    pip_outdated_raw = subprocess.check_output(
        [sys.executable, "-m", "pip", "list", "--outdated", "--format=json"],
        text=True,
    )
    pip_outdated = json.loads(pip_outdated_raw)
    latest_map = {pkg["name"].lower(): pkg["latest_version"] for pkg in pip_outdated}

    rows = []
    for package in requirements:
        current = "unknown"
        latest = "unknown"
        try:
            show_output = subprocess.check_output(
                [sys.executable, "-m", "pip", "show", package], text=True
            )
            for output_line in show_output.splitlines():
                if output_line.startswith("Version:"):
                    current = output_line.split(":", 1)[1].strip()
                    break
            latest = latest_map.get(package.lower(), current)
        except subprocess.CalledProcessError:
            pass

        rows.append([package, current, latest])

    return rows


if __name__ == '__main__':
    test_data = pd.read_csv(os.path.join(test_data_path, "testdata.csv"))
    print(model_predictions(test_data))
    print(dataframe_summary())
    print(missing_data())
    print(execution_time())
    print(outdated_packages_list())
