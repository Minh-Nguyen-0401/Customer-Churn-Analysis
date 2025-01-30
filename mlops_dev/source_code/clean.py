import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
import re
class Cleaner:
    def __init__(self):
        self.mode_imputer = SimpleImputer(strategy='most_frequent',missing_values=np.nan)
    
    def clean_data(self, df):

        # impute rows with missing values
        if df.isnull().values.any():
            missing_idx = df.isnull().any(axis=1)
            df.loc[missing_idx, :] = pd.DataFrame(
                self.mode_imputer.fit_transform(df.loc[missing_idx, :]),
                index=df.loc[missing_idx, :].index,
                columns=df.loc[missing_idx, :].columns
            )
            print(f"Missing values imputed")
        else:
            print(f"No missing values found in original data")

        # feature engineering
        df.columns = [re.sub(r'(?<=[a-z])([A-Z])', r'_\1', i).replace(' ', '_').lower() for i in df.columns]
        df = df.drop(['state', 'area_code'], axis=1)
        df = df.assign(total_minutes=df['total_day_minutes'] + df['total_eve_minutes'] + df['total_night_minutes'])
        df = df.assign(total_calls=df['total_day_calls'] + df['total_eve_calls'] + df['total_night_calls'])
        df = df.assign(total_charge=df['total_day_charge'] + df['total_eve_charge'] + df['total_night_charge'])

        df["intl_charge_per_min"] = df["total_intl_charge"] / df["total_intl_minutes"]

        df["intl_charge_per_min_cate"] = df["intl_charge_per_min"].apply(lambda x: 0 if x <= 0.269 else 1 if x <= 0.2705 else 2).astype(str)

        df["daytime_charge_per_min"] = df["total_day_charge"] / df["total_day_minutes"]
        df["avg_charge_per_acc_day"] = df["total_charge"] / df["account_length"]

        df["voicemail_engagement_lvl"] = df["number_vmail_messages"].apply(lambda x: 0 if x <= 20 else 1 if x <= 35 else 2 if x <= 45 else 3).astype(str)

        df["customer_service_freq_lvl"] = df["customer_service_calls"].apply(lambda x: 0 if x <= 3 else 1 if x <= 6 else 2).astype(str)

        # redefine label's classes
        df['churn'] = df['churn'].map({False: 0, True: 1}).astype(int)

        print(f"Missing values after feature engineering: {df.isna().sum().sum()}")
        return df


