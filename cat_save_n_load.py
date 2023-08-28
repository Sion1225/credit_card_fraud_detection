import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from datetime import datetime
import catboost
import optuna

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

# Train Model with optimal hyper-parameter set
''' Optimal hyper-parameter set
'iterations': 1225, 
'learning_rate': 0.014952415834073335, 
'depth': 15, 
'l2_leaf_reg': 0.8372219126237671, 
'border_count': 321, 
'bagging_temperature': 0.9272616942457109, 
'random_strength': 3.2173100407588158
F1 Score: 0.6143405134257893
'''
model = catboost.CatBoostClassifier(
    iterations=1225, 
    learning_rate=0.014952415834073335, 
    depth=15, 
    l2_leaf_reg=0.8372219126237671,
    border_count=321,
    bagging_temperature=0.9272616942457109,
    random_strength=3.2173100407588158,

    eval_metric="F1",
    boosting_type="Plain", # Ordered or Plain
    cat_features=["user_id", "card_id", "errors?", "merchant_id", "merchant_city", "merchant_state", "mcc", "use_chip", "zip_1"],
    nan_mode="Forbidden",
    scale_pos_weight=scale_pos_weight,
    custom_metric=["Logloss"],
    task_type="GPU"
    )
model.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=100, verbose=1)

# Predict & Validate
y_pred = model.predict(X_test)
model_metric = f1_score(y_test, y_pred)
print(model_metric)

# Save Model
model.save_model("catboostdb/catboost_model_4_" + datetime.now().strftime("%Y%m%d-%H%M%S") + ".cbm")