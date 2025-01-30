import logging
import yaml
# import mlflow
# import mlflow.sklearn
from source_code.ingest import Ingestion
from source_code.clean import Cleaner
from source_code.preprocess import CustomPreprocessor
from source_code.train import Trainer
from source_code.predict import Predictor
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings("ignore")

logging.basicConfig(
    filename='dev_model.log', 
    filemode='w', 
    level=logging.INFO, 
    format='%(asctime)s:%(levelname)s:%(message)s')

def main():
    # Load data
    ingestion = Ingestion()
    train, test = ingestion.load_data()
    logging.info("Data loaded successfully")

    # Clean data
    cleaner = Cleaner()
    train = cleaner.clean_data(train)
    test = cleaner.clean_data(test)
    logging.info("Data cleaned successfully")

    # Preprocess data & Train model
    trainer = Trainer()
    X_train, y_train = trainer.feature_target_split(train, 'churn')
    
    hypertune_req = input("Do you want to hypertune the model? (y/n): ")
    if hypertune_req.lower() == 'y':
        trainer.hypertune_model(X_train, y_train)
    trainer.train_model(X_train, y_train)
    trainer.save_model()
    logging.info("Model trained successfully")

    # Predict & Evaluate
    predictor = Predictor()
    X_test, y_test = predictor.feature_target_split(test, 'churn')
    accuracy, class_report, roc_auc_score = predictor.evaluate_model(X_test, y_test)
    logging.info("Model evaluation completed successfully")
    
    # Print evaluation results
    print("\n============= Model Evaluation Results ==============")
    print(f"Model: {trainer.model_name}")
    print(f"Accuracy Score: {accuracy:.4f}, ROC AUC Score: {roc_auc_score:.4f}")
    print(f"\n{class_report}")
    print("=====================================================\n")

if __name__ == "__main__":
    main()