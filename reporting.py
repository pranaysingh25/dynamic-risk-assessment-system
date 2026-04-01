import pickle
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from diagnostics import model_predictions



###############Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 




##############Function for reporting
def score_model():
    #calculate a confusion matrix using the test data and the deployed model
    #write the confusion matrix to the workspace
    test_data = pd.read_csv(os.path.join(config["test_data_path"], "testdata.csv"))
    y_true = test_data["exited"]
    y_pred = model_predictions(test_data)

    cm = metrics.confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[0, 1], yticklabels=[0, 1])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")

    os.makedirs(config["output_model_path"], exist_ok=True)
    output_path = os.path.join(config["output_model_path"], "confusionmatrix.png")
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()

    return cm





if __name__ == '__main__':
    score_model()
