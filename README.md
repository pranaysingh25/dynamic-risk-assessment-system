# Project Steps Overview

## 5 steps of the project:

<b>1. Data ingestion</b>: Automatically check a database for new data that can be used for model training. Compile all training data to a training dataset and save it to persistent storage. Write metrics related to the completed data ingestion tasks to persistent storage.
<b>2. Training, scoring, and deploying</b>: Write scripts that train an ML model that predicts attrition risk, and score the model. Write the model and the scoring metrics to persistent storage.
<b>3. Diagnostics</b>: Determine and save summary statistics related to a dataset. Time the performance of model training and scoring scripts. Check for dependency changes and package updates.
<b>4. Reporting</b>: Automatically generate plots and documents that report on model metrics. Provide an API endpoint that can return model predictions and metrics.
<b>5. Process Automation</b>: Create a script and cron job that automatically run all previous steps at regular intervals.