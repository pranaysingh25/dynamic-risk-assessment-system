import pandas as pd
import numpy as np
import os
import json
from datetime import datetime




#############Load config.json and get input and output paths
with open('config.json','r') as f:
    config = json.load(f) 

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']



#############Function for data ingestion
def merge_multiple_dataframe():
    #check for datasets, compile them together, and write to an output file
    csv_files = sorted(
        [file for file in os.listdir(input_folder_path) if file.endswith(".csv")]
    )

    dataframes = []
    for file in csv_files:
        file_path = os.path.join(input_folder_path, file)
        dataframes.append(pd.read_csv(file_path))

    if dataframes:
        final_df = pd.concat(dataframes, ignore_index=True).drop_duplicates()
    else:
        final_df = pd.DataFrame()

    os.makedirs(output_folder_path, exist_ok=True)

    final_data_path = os.path.join(output_folder_path, "finaldata.csv")
    final_df.to_csv(final_data_path, index=False)

    ingested_files_path = os.path.join(output_folder_path, "ingestedfiles.txt")
    with open(ingested_files_path, "w") as f:
        f.write(str(csv_files))

    return final_df



if __name__ == '__main__':
    merge_multiple_dataframe()
