import pandas as pd
import yaml

class Ingestion:
    def __init__(self):
        self.config = self.load_config()
    
    def load_config(self):
        with open("config.yml", "r") as f:
            config = yaml.safe_load(f)
        return config
    
    def load_data(self):
        train_path = self.config["data"]["train_path"]
        test_path = self.config["data"]["test_path"]
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        return train_df, test_df