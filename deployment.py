from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
import shutil
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json



##################Load config.json and correct path variable
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
model_path = os.path.join(config["output_model_path"])
prod_deployment_path = os.path.join(config['prod_deployment_path']) 


####################function for deployment
def store_model_into_pickle(model=None):
    #copy the latest pickle file, the latestscore.txt value, and the ingestfiles.txt file into the deployment directory
    os.makedirs(prod_deployment_path, exist_ok=True)

    files_to_copy = [
        os.path.join(model_path, "trainedmodel.pkl"),
        os.path.join(model_path, "latestscore.txt"),
        os.path.join(dataset_csv_path, "ingestedfiles.txt"),
    ]

    for source_file in files_to_copy:
        shutil.copy2(source_file, prod_deployment_path)


if __name__ == "__main__":
    store_model_into_pickle()

