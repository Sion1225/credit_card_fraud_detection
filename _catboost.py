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


# Define Objective function
def Objective(trial):
    # Set Hyper-parameter bounds
    param = {
        'iterations': trial.suggest_int('iterations', 600, 1500, step=25),
        'learning_rate': trial.suggest_float('learning_rate', 5e-3, 0.05),
        'depth': trial.suggest_int('depth', 7, 15),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 0.4, 1),
        'border_count': trial.suggest_int('border_count', 250, 450),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0, 5),
        "random_strength": trial.suggest_float("random_strength", 0, 4),
        'eval_metric': 'F1',
        "boosting_type": "Plain", # Ordered or Plain
        #"rsm": trial.suggest_float("rsm", 0.2, 1),

        "cat_features": ["user_id","card_id","errors?","merchant_id","merchant_city","merchant_state","mcc","use_chip","zip_1"],
        "nan_mode": "Forbidden",
        'scale_pos_weight': scale_pos_weight,
        'custom_metric': ['Logloss'],
        'task_type': 'GPU'
    }

    # Note Hyperparameter set
    with open("catboostdb/CatBoost_Hyper_4.txt", 'a') as f:
        f.write(str(param) + '\n')

    # try 3 times
    all_scores = []
    for _ in range(3):
        # Build CatBoost Classifier and Training
        model = catboost.CatBoostClassifier(**param)
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=100, verbose=0)

        # Predict & Validate
        y_pred = model.predict(X_test)
        model_metric = f1_score(y_test, y_pred)

        # Append Metric
        all_scores.append(model_metric)

    # Note Metric
    with open("catboostdb/CatBoost_Hyper_4.txt", 'a') as f:
        f.write(f"F1 Score: {np.mean(all_scores)} \n\n")

    return np.mean(all_scores)

# Create Optuna sampler and study object
sampler = optuna.samplers.TPESampler(n_startup_trials=40)
study = optuna.create_study(sampler=sampler, 
    study_name="catboost_for_card_fraud_4", 
    direction="maximize", 
    storage="sqlite:///catboostdb/4.db", 
    load_if_exists=True)
study.optimize(Objective, n_trials=440, n_jobs=1)

# Print best hyper-parameter set
with open("catboostdb/CatBoost_Hyper_4.txt",'a') as f:
    f.write(f"Best Hyper-parameter set: \n{study.best_params}\n")
    f.write(f"Best value: {study.best_value}")

print(f"Best Hyper-parameter set: \n{study.best_params}\n")
print(f"Best value: {study.best_value}")
