import logging
import yaml
import mlflow
import mlflow.sklearn
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

    with open("config.yml", "r") as f:
        config = yaml.safe_load(f)
    mlflow.set_experiment("Model Training Experiment")

    with mlflow.start_run(run_name="ndm_bat's Churn Model Development Run") as run:
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

        # Tags
        mlflow.set_tag("Model developer", "minhthemeanie")
        mlflow.set_tag("preprocessing","PowerTransformer, MinMaxScaler, OneHotEncoder, SimpleImputer, SMOTE")

        # Log metrics
        model_params = config['model']['params']
        mlflow.log_params(model_params)

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("roc_auc", roc_auc_score)
        mlflow.log_metric("precision", class_report['1']['precision'])
        mlflow.log_metric("recall", class_report['1']['recall'])
        mlflow.log_metric("f1", class_report['1']['f1-score'])

        # log plot figs
        import os
        roc_curve_path = os.path.join(predictor.config['model']['viz_path'], 'roc_curve.png')
        confusion_matrix_path = os.path.join(predictor.config['model']['viz_path'], 'confusion_matrix.png')

        mlflow.log_artifact(roc_curve_path)
        mlflow.log_artifact(confusion_matrix_path)

        mlflow.sklearn.log_model(trainer.pipeline, "model")

        # Register the model
        model_name = "churn_model"
        model_url = f"runs:/{mlflow.active_run().info.run_id}/model"
        mlflow.register_model(model_uri=model_url, name=model_name)

        logging.info("MLflow tracking registered successfully")

        logging.info(f"\nAccuracy: {accuracy}\
                    \nROC AUC Score: {roc_auc_score}\
                    \n{class_report}")
        
        # Print evaluation results
        print("\n============= Model Evaluation Results ==============")
        print(f"Model: {trainer.model_name}")
        print(f"Accuracy Score: {accuracy:.4f}, ROC AUC Score: {roc_auc_score:.4f}")
        print(f"\n{class_report}")
        print("=====================================================\n")

if __name__ == "__main__":
    main()