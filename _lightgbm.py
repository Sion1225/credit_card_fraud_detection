import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
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

# Create LightGBM Dataset
dtrain = lgb.Dataset(data=X_train, label=y_train, categorical_feature='auto')

# Define Objective function for Optuna
def Objective(trial):
    param = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 1.2, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 500, 1500, step=25),
        "max_depth": trial.suggest_int("max_depth", 5, 15),
        "subsample": trial.suggest_float("subsample", 0.5, 1, step=0.05),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 201, step=5),
        "reg_lambda": trial.suggest_float('lambda', 1e-3, 10.0, log=True),
        "reg_alpha": trial.suggest_float('alpha', 1e-3, 10.0, log=True),
        "scale_pos_weight": scale_pos_weight,
        "device": "gpu"
    }
    
    # Additional logging, if needed
    with open("lightgbmdb/LightGBM_Hyper_1.txt", 'a') as f:
        f.write(str(param) + '\n')
    
    all_scores = []
    for _ in range(3):
        model = lgb.train(param, dtrain, valid_sets=[dtrain], early_stopping_rounds=100, verbose_eval=True)
        y_pred = np.round(model.predict(X_test))
        model_metric = f1_score(y_test, y_pred)
        all_scores.append(model_metric)
        
    with open("lightgbmdb/LightGBM_Hyper_1.txt", 'a') as f:
        f.write(f"F1 Score: {np.mean(all_scores)} \n\n")
        
    return np.mean(all_scores)


# Create Optuna sampler and study object
sampler = optuna.samplers.TPESampler(n_startup_trials=50)
study = optuna.create_study(sampler=sampler, 
                            study_name="lightgbm_for_card_fraud_1", 
                            direction="maximize", 
                            storage="sqlite:///lightgbmdb/1.db", 
                            load_if_exists=True)
study.optimize(Objective, n_trials=550, n_jobs=1)

# Print best hyper-parameter set
with open("lightgbmdb/LightGBM_Hyper_1.txt",'a') as f:
    f.write(f"Best Hyper-parameter set: \n{study.best_params}\n")
    f.write(f"Best value: {study.best_value}")

print(f"Best Hyper-parameter set: \n{study.best_params}\n")
print(f"Best value: {study.best_value}")
