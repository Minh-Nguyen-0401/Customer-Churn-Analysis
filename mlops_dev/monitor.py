import joblib
import pandas as pd
from source_code.clean import Cleaner
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset, TargetDriftPreset
from evidently import ColumnMapping
import warnings
warnings.filterwarnings("ignore")

import mlflow
import yaml
import numpy as np
from mlflow.tracking import MlflowClient

#Check model info
# client = MlflowClient()
# model_info = client.get_model_version(name="churn_model", version="4")
# print(f"Model version 4 stored at: {model_info.source}")

model_name = "churn_model"
model_version = 4 

print(mlflow.get_tracking_uri())
model = mlflow.sklearn.load_model(model_uri=f"models:/{model_name}/{model_version}")


with open("config.yml", "r") as f:
    config = yaml.safe_load(f)

reference = pd.read_csv(config['data']['train_path'])
current = pd.read_csv(config['data']['test_path'])

cleaner = Cleaner()
reference = cleaner.clean_data(reference)
reference['prediction'] = model.predict(reference.drop(columns=["churn"],axis=1))

current = cleaner.clean_data(current)
current['prediction'] = model.predict(current.drop(columns=["churn"],axis=1))

target = "churn"
prediction = 'prediction'
numerical_features = reference.drop(columns=["churn","prediction"],axis=1).select_dtypes(include=np.number).columns.tolist()
categorical_features = [col for col in reference.columns.tolist() if col not in numerical_features and col != "churn" and col != "prediction"]
column_mapping = ColumnMapping()

column_mapping.target = target
column_mapping.prediction = prediction
column_mapping.numerical_features = numerical_features
column_mapping.categorical_features = categorical_features

data_drift_report = Report(metrics=[
    DataDriftPreset(),
    DataQualityPreset(),
    TargetDriftPreset()
])

data_drift_report.run(reference_data=reference, current_data=current, column_mapping=column_mapping)
data_drift_report
# data_drift_report.json()
data_drift_report.save_html("test_drift.html")