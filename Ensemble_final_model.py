import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import xgboost
import tensorflow as tf
import f_score_metrics
import optuna
from kerastuner.tuners import BayesianOptimization

'''
# Read "train.csv" file
df = pd.read_csv("DataSet/train.csv")

# Splitting the 'is_fraud?' column
labels = df["is_fraud?"].copy().to_numpy()
labels = labels.astype(int)
np.savetxt('ensembledb/labels.csv', labels, delimiter=',', fmt='%d')

# Split Datas for train & test
#y_train, y_test = train_test_split(labels, test_size=0.1, random_state=1225)
#np.savetxt('ensembledb/y_train.csv', y_train, delimiter=',', fmt='%d')
#np.savetxt('ensembledb/y_test.csv', y_test, delimiter=',', fmt='%d')
'''

# Read "X" file
X_train = pd.read_csv("ensembledb/X_train_2.csv")
X_test = pd.read_csv("ensembledb/X_test_2.csv")
X_all = pd.read_csv("ensembledb/train_all.csv")
X_submit = pd.read_csv("ensembledb/test_2.csv")

# Make output
output = pd.DataFrame(index=X_submit["index"])

# Drop index
X_train = X_train.set_index(X_train.columns[0])
X_test = X_test.set_index(X_test.columns[0])
X_all = X_all.set_index(X_all.columns[0])
X_submit = X_submit.set_index(X_submit.columns[0])

print(X_all.head(5))

# Read "y" file
y_train = pd.read_csv("ensembledb/y_train.csv", header=None)
y_test = pd.read_csv("ensembledb/y_test.csv", header=None)
labels = pd.read_csv("ensembledb/labels.csv", header=None)
y_train = y_train.to_numpy()
y_test = y_test.to_numpy()
labels = labels.to_numpy()
y_train = y_train.flatten()
y_test = y_test.flatten()
labels = labels.flatten()
print(labels[:3])

print(len(X_train))
print(len(y_train))
print(len(X_test))
print(len(y_test))

# Count fraud or not
num_not_fraud = np.count_nonzero(y_train == 0)
num_fraud = np.count_nonzero(y_train == 1)

scale_pos_weight = num_not_fraud / num_fraud # scale_pos_weight = number of negative instances / number of positive instances

total_samples = len(y_train)

class_weight = {
    0: total_samples / (2 * num_not_fraud),
    1: total_samples / (2 * num_fraud)
}

# DMatrix
#dtrain = xgboost.DMatrix(data=X_train, label=y_train)
#dtest = xgboost.DMatrix(data=X_test, label=y_test)


# ====================================================================================================================
# XGBoost
'''
# Define Objective function
def Objective(trial):
    # Set Hyper-parameter bounds
    param = {
        "booster": "gbtree",
        "objective": "binary:logistic",
        "eval_metric": trial.suggest_categorical("eval_metric", ["logloss","error"]),

        "learning_rate": trial.suggest_float("learning_rate", 0.2, 10, log=True),
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
    with open("ensembledb/XGBoost_Hyper_4.txt", 'a') as f:
        f.write(str(param) + '\n')

    # try n times
    all_scores = []
    for _ in range(1):
        # Build XGBoost Classifier and Training
        model = xgboost.XGBClassifier(**param, early_stopping_rounds=100)
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

        # Predict & Validate
        y_pred = model.predict(X_test)
        model_metric = f1_score(y_test, y_pred)

        # Append Metric
        all_scores.append(model_metric)

    # Note Metric
    with open("ensembledb/XGBoost_Hyper_4.txt", 'a') as f:
        f.write(f"F1 Score: {np.mean(all_scores)} \n\n")

    return np.mean(all_scores)


# Create Optuna sampler and study object
sampler = optuna.samplers.TPESampler(n_startup_trials=30)
study = optuna.create_study(sampler=sampler, 
    study_name="final_xgboost4_fullmodels", 
    direction="maximize", 
    storage="sqlite:///ensembledb/1.db", 
    load_if_exists=True)
study.optimize(Objective, n_trials=330, n_jobs=1)

# Print best hyper-parameter set
with open("ensembledb/XGBoost_Hyper_4.txt",'a') as f:
    f.write(f"Best Hyper-parameter set: \n{study.best_params}\n")
    f.write(f"Best value: {study.best_value}")

print(f"Best Hyper-parameter set: \n{study.best_params}\n")
print(f"Best value: {study.best_value}")
'''

'''
{'eval_metric': 'error', 'learning_rate': 1.4741585073646246, 
'n_estimators': 1420, 'max_depth': 2, 'subsample': 0.7, 
'colsample_bytree': 0.85340325461568, 'gamma': 0.04883309948823751, 
'min_child_weight': 41, 'lambda': 7.525474790541298, 'alpha': 9.999167453761657}
# Best value: 0.8947139753801594
'''

'''
# Train Final XGBoost
# Build XGBoost Classifier and Training
param = {
    'eval_metric': 'error', 'learning_rate': 1.4741585073646246, 
    'n_estimators': 1000, 'max_depth': 2, 'subsample': 0.7, 'colsample_bytree': 0.85340325461568, 
    'gamma': 0.04883309948823751, 'min_child_weight': 41, 'lambda': 7.525474790541298, 'alpha': 9.999167453761657
    }
model = xgboost.XGBClassifier(**param, early_stopping_rounds=100)
model.fit(X_all, labels, eval_set=[(X_test, y_test)], verbose=1)

# Predict & Validate
y_pred = model.predict(X_test)
model_metric = f1_score(y_test, y_pred)
print(model_metric)

# Predict for submit
y_pred = model.predict(X_submit)
output["ensemble"] = y_pred

# Output to CSV
output.to_csv("ensembledb/submit2_2.csv", header=False)
'''

# ====================================================================================================================
# NN

# Define Objective function
def Objective(trial):
    # Set Hyper-parameter bounds
    param = {
        "units1": trial.suggest_int("units1", 3, 500),
        "kernel_l2_lambda1": trial.suggest_float("kernel_l2_lambda1", 1e-4, 1, log=True),
        "activity_l2_lambda1": trial.suggest_float("activity_l2_lambda1", 1e-4, 1, log=True),
        "dropout_rate1": trial.suggest_float("dropout_rate1", 0, 0.5, step=0.05),

        "units2": trial.suggest_int("units2", 3, 500),
        "kernel_l2_lambda2": trial.suggest_float("kernel_l2_lambda2", 1e-4, 1, log=True),
        "activity_l2_lambda2": trial.suggest_float("activity_l2_lambda2", 1e-4, 1, log=True),
        "dropout_rate2": trial.suggest_float("dropout_rate2", 0, 0.5, step=0.05),

        "batch_size": trial.suggest_int("batch_size", 32, 500),
        "learning_rate": trial.suggest_float("lr", 1e-4, 1, log=True),
        "scale_pos_weight": scale_pos_weight
    }

    # Note Hyperparameter set
    with open("ensembledb/FFNN_Hyper_1.txt", 'a') as f:
        f.write(str(param) + '\n')

    units1 = param["units1"]
    kernel_l2_lambda1 = param["kernel_l2_lambda1"]
    activity_l2_lambda1 = param["activity_l2_lambda1"]
    dropout_rate1 = param["dropout_rate1"]
    units2 = param["units2"]
    kernel_l2_lambda2 = param["kernel_l2_lambda2"]
    activity_l2_lambda2 = param["activity_l2_lambda2"]
    dropout_rate2 = param["dropout_rate2"]
    batch_size = param["batch_size"]
    lr = param["learning_rate"]

    # try n times
    all_scores = []
    for _ in range(1):
        # Build FFNN and Training
        inputs = tf.keras.Input(shape=(3,))
        x = tf.keras.layers.Dense(
            units=units1, kernel_regularizer=tf.keras.regularizers.L2(kernel_l2_lambda1), 
            activity_regularizer=tf.keras.regularizers.L2(activity_l2_lambda1), activation="relu", kernel_initializer="he_normal")(inputs)
        x = tf.keras.layers.Dropout(dropout_rate1)(x)
        x = tf.keras.layers.Dense(
            units=units2, kernel_regularizer=tf.keras.regularizers.L2(kernel_l2_lambda2), 
            activity_regularizer=tf.keras.regularizers.L2(activity_l2_lambda2), activation="relu", kernel_initializer="he_normal")(x)
        x = tf.keras.layers.Dropout(dropout_rate2)(x)
        outputs = tf.keras.layers.Dense(units=1, activation="sigmoid")(x)

        model = tf.keras.Model(inputs, outputs)


        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss="binary_crossentropy", metrics=[f_score_metrics.F1Score()])
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, callbacks=[early_stopping], class_weight=class_weight)

        # Predict & Validate
        y_pred = model.predict(X_test)
        model_metric = f1_score(y_test, y_pred)

        # Append Metric
        all_scores.append(model_metric)

    # Note Metric
    with open("ensembledb/FFNN_Hyper_1.txt", 'a') as f:
        f.write(f"F1 Score: {np.mean(all_scores)} \n\n")

    return np.mean(all_scores)


# Create Optuna sampler and study object
sampler = optuna.samplers.TPESampler(n_startup_trials=30)
study = optuna.create_study(sampler=sampler, 
    study_name="final_FFNN1_fullmodels", 
    direction="maximize", 
    storage="sqlite:///ensembledb/1.db", 
    load_if_exists=True)
study.optimize(Objective, n_trials=330, n_jobs=1)

# Print best hyper-parameter set
with open("ensembledb/FFNN_Hyper_1.txt",'a') as f:
    f.write(f"Best Hyper-parameter set: \n{study.best_params}\n")
    f.write(f"Best value: {study.best_value}")

print(f"Best Hyper-parameter set: \n{study.best_params}\n")
print(f"Best value: {study.best_value}")