import os
import joblib
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay
from source_code.train import Trainer
import matplotlib.pyplot as plt
import seaborn as sns

class Predictor:
    def __init__(self):
        self.model_path = self.load_config()['model']['store_path']
        self.model_name = self.load_config()['model']['name']
        self.pipeline = self.load_model()
        self.trainer = Trainer()
        self.feature_target_split = self.trainer.feature_target_split
        self.config = self.load_config()

    def load_config(self):
        import yaml
        with open('config.yml', 'r') as config_file:
            return yaml.safe_load(config_file)
        
    def load_model(self):
        model_file_path = os.path.join(self.model_path, self.model_name + '.joblib')
        return joblib.load(model_file_path)

    def evaluate_model(self, X_test, y_test):
        y_pred = self.pipeline.predict(X_test)
        y_pred_proba = self.pipeline.predict_proba(X_test)[:,1]
        accuracy = accuracy_score(y_test, y_pred)
        class_report = classification_report(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred)
        
        # export metrics figs
        viz_path = self.config['model']['viz_path']
        if not os.path.exists(viz_path):
            os.makedirs(viz_path)

        # ROC curve
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(viz_path, 'roc_curve.png'))
        plt.close()

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, cmap='Blues', fmt='d')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.savefig(os.path.join(viz_path, 'confusion_matrix.png'))
        plt.close()

        return accuracy, class_report, roc_auc