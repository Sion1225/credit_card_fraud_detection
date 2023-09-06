import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import lightgbm as lgb
from datetime import datetime

# Read "train.csv" file
df = pd.read_csv("DataSet/train.csv")

# Splitting the 'is_fraud?' column
labels = df["is_fraud?"].copy().to_numpy()
labels = labels.astype(int)
df = df.drop("is_fraud?", axis=1)
df = df.set_index(df.columns[0])

# Delete $ symbol from amount column
df['amount'] = df['amount'].str.replace('$', '').astype(float)

# Split "zip" by units
df["zip_1"] = df["zip"] // 10000
df["zip_2"] = (df["zip"] - df["zip_1"]) // 100
df["zip_4"] = df["zip"] % 100

# Drop "zip"
df = df.drop("zip", axis=1)

# Replace NaN with -1
df = df.fillna(-1)

# Change float64 with int64
df["amount"] = round(df["amount"] * 100)
df["amount"] = df["amount"].astype("int64")
df["zip_2"] = df["zip_2"].astype("int64")
df["zip_4"] = df["zip_4"].astype("int64")

df["merchant_id"] = df["merchant_id"].astype("category")
df["mcc"] = df["mcc"].astype("category")
df["merchant_city"] = df["merchant_city"].astype("category")
df["merchant_state"] = df["merchant_state"].astype("category")
df["errors?"] = df["errors?"].astype("category")
df["use_chip"] = df["use_chip"].astype("category")
df["user_id"] = df["user_id"].astype("category")
df["card_id"] = df["card_id"].astype("category")
df["zip_1"] = df["zip_1"].astype("int64")
df["zip_1"] = df["zip_1"].astype("category")


# Create additional data
# User average amount
user_avg_amount = df.groupby("user_id")["amount"].mean().reset_index()
user_avg_amount.columns = ['user_id', 'user_avg_amount']

# Merchant average amount
merchant_avg_amount = df.groupby("merchant_id")["amount"].mean().reset_index()
merchant_avg_amount.columns = ["merchant_id", "merchant_avg_amount"]

# Add to Original Dataset
df = pd.merge(df, user_avg_amount, on="user_id", how="left")
df = pd.merge(df, merchant_avg_amount, on="merchant_id", how="left")


# Print Data sample
print(labels[:5])
print(df.head(10))

# Validate
print(df.dtypes)

# Split Datas for train & test
X_train, X_test, y_train, y_test = train_test_split(df, labels, test_size=0.1, random_state=1225)

# Count fraud or not
num_not_fraud = np.count_nonzero(y_train == 0)
num_fraud = np.count_nonzero(y_train == 1)
scale_pos_weight = num_not_fraud / num_fraud

# Create LightGBM Dataset
dtrain = lgb.Dataset(data=X_train, label=y_train, categorical_feature='auto')
dtest = lgb.Dataset(data=X_test, label=y_test, categorical_feature="auto")


# Set object with optimal hyper-parameter set
'''Optimal hyper-parameter set
'learning_rate': 0.40800370142469505, 
'n_estimators': 1425, 
'max_depth': 13, 
'min_child_weight': 1, 
'lambda': 2.217118681010151, 
'alpha': 0.013291781490949256, 
'feature_fraction': 0.867512174769522, 
'bagging_fraction': 1.0, 
'bagging_freq': 3, 
'min_split_gain': 0.2684006150772534
F1 Score: 0.5679989784191036
'''
params ={
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    'learning_rate': 0.40800370142469505, 
    'max_depth': 13, 
    'min_child_weight': 1, 
    'lambda': 2.217118681010151, 
    'alpha': 0.013291781490949256, 
    'feature_fraction': 0.867512174769522, 
    'bagging_fraction': 1.0, 
    'bagging_freq': 3, 
    'min_split_gain': 0.2684006150772534,
    "scale_pos_weight": scale_pos_weight,
    "device": "cpu"
}
callbacks = [lgb.early_stopping(stopping_rounds=100, first_metric_only=False, verbose=True)]
model = lgb.train(params, dtrain, num_boost_round=1425, valid_sets=[dtest, dtrain], callbacks=callbacks)
y_pred = np.round(model.predict(X_test))
model_metric = f1_score(y_test, y_pred)
print(model_metric)

# Save Model
model.save_model("lightgbmdb/lightgbm_model_1_" + str(model_metric) + "_" + datetime.now().strftime("%Y%m%d-%H%M%S") + ".txt")

# Non data seperated model
dtest = lgb.Dataset(data=df, label=labels, categorical_feature='auto')
# Train model
model = lgb.train(params, dtest, num_boost_round=1425, valid_sets=[dtest], callbacks=callbacks)

# Save Model
model.save_model("lightgbmdb/lightgbm_full_data_model_1_" + str(model_metric) + "_" + datetime.now().strftime("%Y%m%d-%H%M%S") + ".txt")