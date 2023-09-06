import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from category_encoders import BinaryEncoder
from sklearn.ensemble import RandomForestClassifier
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


# BinaryEncoding
binary_encoder = BinaryEncoder(cols=["merchant_id", "mcc", "merchant_city", "merchant_state", "errors?", "use_chip", "user_id", "card_id", "zip_1"])
df = binary_encoder.fit_transform(df)


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


# Define Objective function for Optuna
def Objective(trial):
    param = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "max_depth": trial.suggest_int("max_depth", 2, 32, log=True),
        "min_samples_split": trial.suggest_float("min_samples_split", 0.1, 1),
        "min_samples_leaf": trial.suggest_float("min_samples_leaf", 0.1, 0.5),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
        "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
        "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"]),
        "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 2, 128, log=True),
        "min_impurity_decrease": trial.suggest_float("min_impurity_decrease", 0.0, 1),
        "random_state": 1225
    }
    
    # Additional logging, if needed
    with open("randomforestdb/Random_2.txt", 'a') as f:
        f.write(str(param) + '\n')
    
    model = RandomForestClassifier(**param)
    model.fit(X_train, y_train)
    
    y_pred = np.round(model.predict(X_test))
    model_metric = f1_score(y_test, y_pred)
        
    with open("randomforestdb/Random_2.txt", 'a') as f:
        f.write(f"F1 Score: {model_metric} \n\n")
        
    return model_metric


# Create Optuna sampler and study object
sampler = optuna.samplers.TPESampler(n_startup_trials=40)
study = optuna.create_study(sampler=sampler, 
                            study_name="random_for_card_fraud_2", 
                            direction="maximize", 
                            storage="sqlite:///randomforestdb/1.db", 
                            load_if_exists=True)
study.optimize(Objective, n_trials=440, n_jobs=1)

# Print best hyper-parameter set
with open("randomforestdb/Random_2.txt",'a') as f:
    f.write(f"Best Hyper-parameter set: \n{study.best_params}\n")
    f.write(f"Best value: {study.best_value}")

print(f"Best Hyper-parameter set: \n{study.best_params}\n")
print(f"Best value: {study.best_value}")