import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from datetime import datetime
import catboost

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

# Set object with optimal hyper-parameter set
''' Optimal hyper-parameter set
'iterations': 700, 
'learning_rate': 0.026067560703260176, 
'depth': 16, 
'l2_leaf_reg': 0.6250112975779198, 
'border_count': 356, 
'bagging_temperature': 0.09747102534316096, 
'random_strength': 2.7071847149006034

Best is trial 129 with value: 0.620253164556962
'''
model = catboost.CatBoostClassifier(
    iterations=700, 
    learning_rate=0.026067560703260176, 
    depth=16, 
    l2_leaf_reg=0.6250112975779198,
    border_count=356,
    bagging_temperature=0.09747102534316096,
    random_strength=2.7071847149006034,

    eval_metric="F1",
    boosting_type="Plain", # Ordered or Plain
    cat_features=["user_id", "card_id", "errors?", "merchant_id", "merchant_city", "merchant_state", "mcc", "use_chip", "zip_1"],
    nan_mode="Forbidden",
    scale_pos_weight=scale_pos_weight,
    custom_metric=["Logloss"],
    task_type="GPU"
    )

# Data Seperated model
# Train model
model.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=100, verbose=1)

# Predict & Validate
y_pred = model.predict(X_test)
model_metric = f1_score(y_test, y_pred)
print(model_metric)

# Save Model
model.save_model("catboostdb/catboost_model_11_" + str(model_metric) + "_" + datetime.now().strftime("%Y%m%d-%H%M%S") + ".cbm")


# Non data seperated model
# Train model
model.fit(df, labels, eval_set=[(X_test, y_test)], early_stopping_rounds=100, verbose=1)

# Save Model
model.save_model("catboostdb/catboost_full_data_model_11_" + str(model_metric) + "_" + datetime.now().strftime("%Y%m%d-%H%M%S") + ".cbm")