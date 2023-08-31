import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from datetime import datetime
import catboost

# Read "train.csv" file
df = pd.read_csv("DataSet/test.csv")

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
print(df.head(10))

# Validate
print(df.dtypes)

# Load model
model = catboost.CatBoostClassifier()
model.load_model("catboostdb/catboost_model_5_0.6143405134257893_20230901-001150.cbm")

# Test loaded model
y_pred = model.predict(df)

# Write on Dataframe
output["ans"] = y_pred

# Output to CSV
output.to_csv("catboostdb/submit2_1.csv", header=False)