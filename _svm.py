import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.svm import SVC
from category_encoders import BinaryEncoder
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

# Print Data sample
print(labels[:5])
print(df.head(10))

# Validate
print(df.dtypes)

# BinaryEncoding
binary_encoder = BinaryEncoder(cols=["merchant_id", "mcc", "merchant_city", "merchant_state", "errors?", "use_chip", "user_id", "card_id", "zip_1"])
df = binary_encoder.fit_transform(df)

# Split Datas for train & test
X_train, X_test, y_train, y_test = train_test_split(df, labels, test_size=0.1, random_state=1225)

# Count fraud or not
num_not_fraud = np.count_nonzero(y_train == 0)
num_fraud = np.count_nonzero(y_train == 1)
scale_pos_weight = num_not_fraud / num_fraud


# Define Objective function for SVM
def Objective(trial):
    # Set Hyper-parameter bounds
    param = {
        'C': trial.suggest_float('C', 1e-3, 1000, log=True),
        'kernel': trial.suggest_categorical('kernel', ['linear', 'rbf', 'poly', 'sigmoid']),
        'degree': trial.suggest_int('degree', 1, 5),
        'gamma': trial.suggest_categorical('gamma', ['scale', 'auto']),
        'coef0': trial.suggest_float('coef0', -1, 1),
        'shrinking': trial.suggest_categorical('shrinking', [True, False]),
        'probability': False,
        'tol': trial.suggest_float('tol', 1e-5, 1e-3, log=True),
        'cache_size': 2000  # Set a large cache size for faster computation
    }
    
    # Note Hyperparameter set
    with open("svmdb/SVM_Hyper_1.txt", 'a') as f:
        f.write(str(param) + '\n')
        
    # Build SVM Classifier and Training
    model = SVC(**param)
    model.fit(X_train, y_train)

    # Predict & Validate
    y_pred = model.predict(X_test)
    model_metric = f1_score(y_test, y_pred)

    # Note Metric
    with open("svmdb/SVM_Hyper_1.txt", 'a') as f:
        f.write(f"F1 Score: {model_metric} \n\n")

    return model_metric

# Create Optuna sampler and study object
sampler = optuna.samplers.TPESampler(n_startup_trials=15)
study = optuna.create_study(sampler=sampler, 
    study_name="svm_for_card_fraud_1", 
    direction="maximize", 
    storage="sqlite:///svmdb/1.db", 
    load_if_exists=True)
study.optimize(Objective, n_trials=150, n_jobs=8)

# Print best hyper-parameter set
with open("svmdb/SVM_Hyper_1.txt",'a') as f:
    f.write(f"Best Hyper-parameter set: \n{study.best_params}\n")
    f.write(f"Best value: {study.best_value}")

print(f"Best Hyper-parameter set: \n{study.best_params}\n")
print(f"Best value: {study.best_value}")
