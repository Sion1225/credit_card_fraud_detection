import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from datetime import datetime
import catboost
import lightgbm as lgb

# Read "train.csv" file
df = pd.read_csv("DataSet/train.csv")

# Splitting the 'is_fraud?' column
labels = df["is_fraud?"].copy().to_numpy()
labels = labels.astype(int)
df = df.drop("is_fraud?", axis=1)
output = pd.DataFrame(index=df["index"])
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


# Print Data sample
print(labels[:5])
print(df.head(5))

# Validate
print(df.dtypes)

# Split Datas for train & test
X_train, X_test, y_train, y_test = train_test_split(df, labels, test_size=0.1, random_state=1225)


# Catboost
def catboost_function(X_test):
    print("===========CatBoost===========")
    # Load model
    model = catboost.CatBoostClassifier()
    model.load_model("catboostdb/catboost_model_4_0.6143405134257893_20230829-004440.cbm")

    # Test loaded model
    y_pred = model.predict(X_test)

    # Validation model 
    print(f1_score(y_pred, y_test))

    # Predict Probability & save to dataframe
    y_pred = model.predict(X_test, prediction_type='Probability')

    print("===============================================")

    return y_pred


# Create additional data
# User average amount
def add_data(df):
    print("Making additional datas")
    user_avg_amount = df.groupby("user_id")["amount"].mean().reset_index()
    user_avg_amount.columns = ['user_id', 'user_avg_amount']

    # Merchant average amount
    merchant_avg_amount = df.groupby("merchant_id")["amount"].mean().reset_index()
    merchant_avg_amount.columns = ["merchant_id", "merchant_avg_amount"]

    # Add to Original Dataset
    df = pd.merge(df, user_avg_amount, on="user_id", how="left")
    df = pd.merge(df, merchant_avg_amount, on="merchant_id", how="left")

    return df


# LightGBM
def lightgbm_fucntion(X_test):
    print("===========LightGBM===========")
    model = lgb.Booster()
    model.load_model('lightgbmdb/lightgbm_model_1_0.565894417014812_20230904-041101.txt')

    # Test loaded model
    y_pred = np.round(model.predict(X_test))

    # Validation model 
    print(f1_score(y_pred, y_test))

    # Predict Probability & save to dataframe
    y_pred = model.predict(X_test)

    print("===============================================")

    return y_pred


# Catboost
y_pred = catboost_function(X_test)
output["catboost"] = y_pred
print(output.head(5))

# Add data
df = add_data(df)
print(df.head(5))

# LightGBM
y_pred = lightgbm_fucntion(X_test)
output["lightgbm"] = y_pred
print(output.head(5))