import os
import joblib
import yaml
import pandas as pd
import numpy as np
from tabulate import tabulate
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler, PowerTransformer
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
import optbinning
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer
from statsmodels.stats.outliers_influence import variance_inflation_factor


class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def calculate_woe_iv(self, X, y, variable, feature_list, IV_list, special_value=None, max_bins=10):
        if str(X.dtype) != "object":
            optb = optbinning.OptimalBinning(name=variable, dtype="numerical", max_n_bins=max_bins)
        else:
            optb = optbinning.OptimalBinning(name=variable, dtype="categorical")

        optb.fit(X, y)

        binning_table = optb.binning_table.build()
        IV = binning_table["IV"].max()

        feature_list.append(variable)
        IV_list.append(IV)

    # Function to calculate IV for all features
    def calc_iv_return_rel_features(self, df, feature_col, target_col, threshold = 0.02):
        feature_list = []
        IV_list = []
        for col in feature_col:
            self.calculate_woe_iv(df[col], df[target_col], col, feature_list, IV_list)
            # print(f"{col} calculated iv done")

        iv_df = pd.DataFrame({"feature": feature_list, "IV": IV_list})
        iv_df = iv_df.sort_values("IV", ascending=False).reset_index(drop=True)
        print(tabulate(iv_df, headers="keys", tablefmt="psql"))
        rel_features = iv_df[iv_df["IV"]>=threshold]["feature"].values.tolist()

        print(f"Selected {len(rel_features)} features with decent IV, including {rel_features}")
        return rel_features
    
    # Function to calculate VIF
    def calculate_vif(self, df, target, threshold = np.inf):
        """Removes features with high VIF (multicollinearity)."""

        num_df = df.select_dtypes(include=np.number).dropna()
        num_df = num_df.drop(target, axis=1) if target in num_df.columns else num_df

        cate_features = [col for col in df.columns if col not in num_df.columns and col != target]

        vif_data = pd.DataFrame()
        vif_data["Feature"] = num_df.columns
        vif_data["VIF"] = [variance_inflation_factor(num_df.values, i) for i in range(num_df.shape[1])]

        # Keep features with VIF < threshold
        selected_num_features = vif_data[vif_data["VIF"] < threshold]["Feature"].tolist()
        dropped_features = vif_data[vif_data["VIF"] >= threshold]["Feature"].tolist()

        print(tabulate(vif_data, headers="keys", tablefmt="psql"))
        print(f"Selected {len(selected_num_features)} features with low VIF, including {selected_num_features}")

        final_features = selected_num_features + cate_features

        print(f"Final {len(final_features)} features: {final_features}")
        return final_features


from sklearn.pipeline import Pipeline

class CustomPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.config = self.load_config()
        self.feature_selector = FeatureSelector()
        self.preprocessor = None
        self.rel_features = None

    def load_config(self):
        with open("config.yml", "r") as f:
            config = yaml.safe_load(f)
        return config

    def fit(self, X, y):
        y_df = pd.DataFrame(y, columns=["churn"])
        df = pd.concat([X, y_df], axis=1)

        # Feature selection
        rel_features_iv = self.feature_selector.calc_iv_return_rel_features(
            df, X.columns, "churn"
        )

        calc_vif_req = self.config["feature_selection"]["use_vif"]
        if calc_vif_req:
            rel_features_vif = self.feature_selector.calculate_vif(
                df[rel_features_iv], "churn"
            )
            self.rel_features = rel_features_vif
        else:
            self.rel_features = rel_features_iv


        num_features = X[self.rel_features].select_dtypes(include=np.number).columns.tolist()
        cate_features = [col for col in self.rel_features if col not in num_features]

        # Build a numeric pipeline
        numeric_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('yeo', PowerTransformer(method='yeo-johnson')),
            ('minmax', MinMaxScaler())
        ])

        # Build ColumnTransformer
        self.preprocessor = ColumnTransformer([
            ('num_pipeline', numeric_pipeline, num_features),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop="first"), cate_features)
        ])

        # Fit the preprocessor
        self.preprocessor.fit(X[self.rel_features])
        return self

    def transform(self, X):
        if self.preprocessor is None:
            raise ValueError("Preprocessor not fitted. Call fit() first.")

        # subset to relevant columns
        X_selected = X[self.rel_features]

        # apply transform
        X_transformed = self.preprocessor.transform(X_selected)

        # get the correct expanded feature names
        feature_names = self._get_feature_names()
        if X_transformed.shape[1] != len(feature_names):
            raise ValueError(
                f"Shape mismatch: {X_transformed.shape[1]} vs. {len(feature_names)} feature names."
            )

        final_X = pd.DataFrame(X_transformed, columns=feature_names)
        return final_X

    def _get_feature_names(self):
        """Derive final feature names from the ColumnTransformer after fit."""

        numeric_cols = self.preprocessor.transformers_[0][2]  

        ohe = self.preprocessor.transformers_[1][1]  
        cat_cols = self.preprocessor.transformers_[1][2]  

        # get expanded ohe column names
        if hasattr(ohe, "get_feature_names_out"):
            cat_feature_names = ohe.get_feature_names_out(cat_cols)
        else:
            cat_feature_names = []

        return list(numeric_cols) + list(cat_feature_names)

    

    