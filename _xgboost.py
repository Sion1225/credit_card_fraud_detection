import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import xgboost
import optuna

# Read "train.csv" file
df = pd.read_csv("DataSet/train.csv")

# Splitting the 'is_fraud?' column
labels = df["is_fraud?"].copy().to_numpy()
labels = labels.astype(int)
df = df.drop("is_fraud?", axis=1)
df = df.set_index(df.columns[0])

# Delete $ simbol from amount column
df['amount'] = df['amount'].str.replace('$', '').astype(float)

# Print Data sample
print(labels[:5])
print(df.head())

# Set Categorical data for XGBoost
df["merchant_id"] = df["merchant_id"].astype("category")
df["mcc"] = df["mcc"].astype("category")
df["merchant_city"] = df["merchant_city"].astype("category")
df["merchant_state"] = df["merchant_state"].astype("category")
df["errors?"] = df["errors?"].astype("category")
df["use_chip"] = df["use_chip"].astype("category")
df["user_id"] = df["user_id"].astype("category")
df["card_id"] = df["card_id"].astype("category")

''' DataSet Comment
결제 주랑 zip code에는 결측치가 있음. 이 결측치들은 대부분 미국이 아닌 해외.
zip과 merchant_state 은 반드시 동시에 NaN 이며, 이 때 결제수단은 온라인.
'''

# Split "zip" by units
df["zip_1"] = df["zip"] // 10000
df["zip_2"] = (df["zip"] - df["zip_1"]) // 100
df["zip_4"] = df["zip"] % 100

df["zip_1"] = df["zip_1"].astype("category")

# Drop "zip"
df = df.drop("zip", axis=1)

# Validate
print(df.dtypes)

# Split Datas for train & test
X_train, X_test, y_train, y_test = train_test_split(df, labels, test_size=0.1, random_state=1225)

# Count fraud or not
num_not_fraud = np.count_nonzero(y_train == 0)
num_fraud = np.count_nonzero(y_train == 1)

scale_pos_weight = num_not_fraud / num_fraud # scale_pos_weight = number of negative instances / number of positive instances

# DMatrix
dtrain = xgboost.DMatrix(data=X_train, label=y_train, enable_categorical=True)


# Define Objective function
def Objective(trial):
    # Set Hyper-parameter bounds
    param = {
        "booster": "gbtree",
        "objective": "binary:logistic",
        "eval_metric": "logloss",

        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 1.2, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 500, 1500, step=25),
        "max_depth": trial.suggest_int("max_depth", 5, 15),
        "subsample": trial.suggest_float("subsample", 0.5, 1, step=0.05),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1),
        "gamma": trial.suggest_float("gamma", 0, 10), # default
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 201, step=5),
        'lambda': trial.suggest_float('lambda', 1e-3, 10.0, log=True),
        'alpha': trial.suggest_float('alpha', 1e-3, 10.0, log=True),

        "device": "cuda",
        "tree_method": "hist", #gpu_
        "scale_pos_weight": scale_pos_weight
    }

    # Note Hyperparameter set
    with open("xgboostdb/XGBoost_Hyper_2.txt", 'a') as f:
        f.write(str(param) + '\n')

    # try 3 times
    all_scores = []
    for _ in range(3):
        # Build XGBoost Classifier and Training
        model = xgboost.XGBClassifier(**param, early_stopping_rounds=100, enable_categorical=True)
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

        # Predict & Validate
        y_pred = model.predict(X_test)
        model_metric = f1_score(y_test, y_pred)

        # Append Metric
        all_scores.append(model_metric)

    # Note Metric
    with open("xgboostdb/XGBoost_Hyper_2.txt", 'a') as f:
        f.write(f"F1 Score: {np.mean(all_scores)} \n\n")

    return np.mean(all_scores)


# Create Optuna sampler and study object
sampler = optuna.samplers.TPESampler(n_startup_trials=50)
study = optuna.create_study(sampler=sampler, 
    study_name="xgboost_for_card_fraud_2", 
    direction="maximize", 
    storage="sqlite:///xgboostdb/2.db", 
    load_if_exists=True)
study.optimize(Objective, n_trials=550, n_jobs=1)

# Print best hyper-parameter set
with open("xgboostdb/XGBoost_Hyper_2.txt",'a') as f:
    f.write(f"Best Hyper-parameter set: \n{study.best_params}\n")
    f.write(f"Best value: {study.best_value}")

print(f"Best Hyper-parameter set: \n{study.best_params}\n")
print(f"Best value: {study.best_value}")
