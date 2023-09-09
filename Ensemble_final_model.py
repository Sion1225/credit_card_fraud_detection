import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import xgboost
import optuna

'''
# Read "train.csv" file
df = pd.read_csv("DataSet/train.csv")

# Splitting the 'is_fraud?' column
labels = df["is_fraud?"].copy().to_numpy()
labels = labels.astype(int)

# Split Datas for train & test
y_train, y_test = train_test_split(labels, test_size=0.1, random_state=1225)
np.savetxt('ensembledb/y_train.csv', y_train, delimiter=',', fmt='%d')
np.savetxt('ensembledb/y_test.csv', y_test, delimiter=',', fmt='%d')
'''

# Read "X" file
X_train = pd.read_csv("ensembledb/X_train.csv")
X_test = pd.read_csv("ensembledb/X_test.csv")

# Drop index
X_train = X_train.set_index(X_train.columns[0])
X_test = X_test.set_index(X_test.columns[0])

print(X_train.head(5))

# Read "y" file
y_train = pd.read_csv("ensembledb/y_train.csv", header=None)
y_test = pd.read_csv("ensembledb/y_test.csv", header=None)
y_train = y_train.to_numpy()
y_test = y_test.to_numpy()
y_train = y_train.flatten()
y_test = y_test.flatten()
print(y_train[:3])

# Count fraud or not
num_not_fraud = np.count_nonzero(y_train == 0)
num_fraud = np.count_nonzero(y_train == 1)

scale_pos_weight = num_not_fraud / num_fraud # scale_pos_weight = number of negative instances / number of positive instances

# DMatrix
dtrain = xgboost.DMatrix(data=X_train, label=y_train)


# ====================================================================================================================
# XGBoost

# Define Objective function
def Objective(trial):
    # Set Hyper-parameter bounds
    param = {
        "booster": "gbtree",
        "objective": "binary:logistic",
        "eval_metric": trial.suggest_categorical("eval_metric", ["logloss","error"]),

        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 2, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 500, 1500, step=20),
        "max_depth": trial.suggest_int("max_depth", 2, 8),
        "subsample": trial.suggest_float("subsample", 0.5, 1, step=0.05),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1),
        "gamma": trial.suggest_float("gamma", 1e-4, 0.05),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 201),
        'lambda': trial.suggest_float('lambda', 0.02, 12.0),
        'alpha': trial.suggest_float('alpha', 0.02, 10.0),

        "device": "cuda",
        "tree_method": "hist",
        "scale_pos_weight": scale_pos_weight
    }

    # Note Hyperparameter set
    with open("ensembledb/XGBoost_Hyper_2.txt", 'a') as f:
        f.write(str(param) + '\n')

    # try n times
    all_scores = []
    for _ in range(1):
        # Build XGBoost Classifier and Training
        model = xgboost.XGBClassifier(**param, early_stopping_rounds=100, enable_categorical=True)
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

        # Predict & Validate
        y_pred = model.predict(X_test)
        model_metric = f1_score(y_test, y_pred)

        # Append Metric
        all_scores.append(model_metric)

    # Note Metric
    with open("ensembledb/XGBoost_Hyper_2.txt", 'a') as f:
        f.write(f"F1 Score: {np.mean(all_scores)} \n\n")

    return np.mean(all_scores)


# Create Optuna sampler and study object
sampler = optuna.samplers.TPESampler(n_startup_trials=30)
study = optuna.create_study(sampler=sampler, 
    study_name="final_xgboost2", 
    direction="maximize", 
    storage="sqlite:///ensembledb/1.db", 
    load_if_exists=True)
study.optimize(Objective, n_trials=330, n_jobs=1)

# Print best hyper-parameter set
with open("ensembledb/XGBoost_Hyper_2.txt",'a') as f:
    f.write(f"Best Hyper-parameter set: \n{study.best_params}\n")
    f.write(f"Best value: {study.best_value}")

print(f"Best Hyper-parameter set: \n{study.best_params}\n")
print(f"Best value: {study.best_value}")