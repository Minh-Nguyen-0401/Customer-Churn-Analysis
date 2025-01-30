import os
import joblib
import yaml
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline 
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from lightgbm import LGBMClassifier
from source_code.preprocess import CustomPreprocessor
from sklearn.model_selection import StratifiedKFold
from skopt import BayesSearchCV
from tabulate import tabulate
from skopt.space import Real, Integer  # If using Bayesian Optimization


class Trainer:
    def __init__(self):
        self.config = self.load_config()
        self.model_name = self.config['model']['name']
        self.model_params = self.config['model']['params']
        self.model_path = self.config['model']['store_path']
        self.hyper_params_space = self.config['model']['hyperparameter_space']
        self.pipeline = self.create_pipeline()

    def load_config(self):
        with open('config.yml', 'r') as config_file:
            return yaml.safe_load(config_file)
        
    def update_params(self, params):
        self.model_params.update(params)

        with open('config.yml', 'w') as config_file:
            yaml.dump(self.config, config_file, default_flow_style=False)
        
        print("\n============= Model Parameters Updated ==============")
        print(self.model_params)
    
    def hypertune_model(self,X_train, y_train, scoring='roc_auc'):
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        self.hyper_params_space = {
            'model__n_estimators': Integer(*self.config['model']['hyperparameter_space']['n_estimators']),
            'model__max_depth': Integer(*self.config['model']['hyperparameter_space']['max_depth']),
            'model__learning_rate': Real(
                self.config['model']['hyperparameter_space']['learning_rate'][0], 
                self.config['model']['hyperparameter_space']['learning_rate'][1], 
                prior='log-uniform'
            ),
            'model__subsample': Real(*self.config['model']['hyperparameter_space']['subsample']),
            'model__colsample_bytree': Real(*self.config['model']['hyperparameter_space']['colsample_bytree']),
            'model__scale_pos_weight': (scale_pos_weight,),
            'model__reg_alpha': Real(*self.config['model']['hyperparameter_space']['reg_alpha']),
            'model__reg_lambda': Real(*self.config['model']['hyperparameter_space']['reg_lambda']),
            'model__min_child_weight': Integer(*self.config['model']['hyperparameter_space']['min_child_weight']),
            'model__verbose': (self.config['model']['hyperparameter_space']['verbose'],),
            'model__random_state': (self.config['model']['hyperparameter_space']['random_state'],),
            'model__criterion': (self.config['model']['hyperparameter_space']['criterion'],)
        }

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        search = BayesSearchCV(self.pipeline, self.hyper_params_space, cv=skf, n_iter=10, random_state=42,
                               n_jobs=-1, verbose=1, scoring=scoring, return_train_score=True)
        search.fit(X_train, y_train)
        train_score_df = pd.DataFrame(search.cv_results_)
        print("\n============= Hyperparameter Tuning Results ==============")
        print(tabulate(train_score_df, headers='keys', tablefmt='psql'))

        best_params = search.best_params_

        self.update_params(best_params)
        

    def create_pipeline(self):
        model_map = {
            'RandomForestClassifier': RandomForestClassifier,
            'GradientBoostingClassifier': GradientBoostingClassifier,
            'DecisionTreeClassifier': DecisionTreeClassifier,
            'LGBMClassifier': LGBMClassifier
        }
        preprocessor = CustomPreprocessor()

        smote = SMOTE(sampling_strategy='minority', random_state=42, k_neighbors=5)
        model = model_map[self.model_name](**self.model_params)

        print("\n==============Model stats==============")
        print(f"{model}")
        
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('smote', smote),
            ('model', model)
        ])
        print("\n==============Model Pipeline==============")
        print(pipeline)
        return pipeline
    
    def feature_target_split(self, df, target_col):
        X = df.drop(target_col, axis=1)
        y = df[target_col]
        return X, y
    
    def train_model(self, X_train, y_train):
        self.pipeline.fit(X_train, y_train)

    def save_model(self):
        file_path = os.path.join(self.model_path, self.model_name + '.joblib')
        joblib.dump(self.pipeline, file_path)