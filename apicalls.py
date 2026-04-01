import requests
import json
import os

# Specify a URL that resolves to your workspace
URL = "http://127.0.0.1:8000"


with open("config.json", "r") as f:
    config = json.load(f)


# Call each API endpoint and store the responses
response1 = requests.post(
    f"{URL}/prediction",
    json={"dataset_path": os.path.join(config["test_data_path"], "testdata.csv")},
)
response2 = requests.get(f"{URL}/scoring")
response3 = requests.get(f"{URL}/summarystats")
response4 = requests.get(f"{URL}/diagnostics")

# combine all API responses
responses = {
    "prediction": response1.json(),
    "scoring": response2.json(),
    "summarystats": response3.json(),
    "diagnostics": response4.json(),
}

# write the responses to your workspace
os.makedirs(config["output_model_path"], exist_ok=True)
output_path = os.path.join(config["output_model_path"], "apireturns.txt")
with open(output_path, "w") as f:
    f.write(json.dumps(responses))