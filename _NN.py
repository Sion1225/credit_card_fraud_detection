import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from category_encoders import BinaryEncoder
import tensorflow as tf
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
print("BinaryEncoding")
category_vars = ["merchant_id", "mcc", "merchant_city", "merchant_state", "errors?", "use_chip", "user_id"]
binary_encoder = BinaryEncoder(cols=category_vars)
df = binary_encoder.fit_transform(df)

# Combine encodered column
print("Combine encodered column")
for var in category_vars:
    cols_to_combine = [col for col in df.columns if col.startswith(var)]
    sub_data = df[cols_to_combine].to_numpy(dtype=str)
    combined = np.apply_along_axis(''.join, axis=1, arr=sub_data)
    df[var] = combined
    df[var] = df[var].astype("int64")
    df.drop(cols_to_combine, axis=1, inplace=True)
    

# Print Data sample
print(labels[:5])
print(df.head(10))

# Validate
print(df.dtypes)

# Split Datas for train & test
X_train, X_test, y_train, y_test = train_test_split(df, labels, test_size=0.1, random_state=1225)

# Shift to tf.data.Dataset
train_dataset = tf.data.Dataset.from_tensor_slices((dict(X_train), y_train))
test_dataset = tf.data.Dataset.from_tensor_slices((dict(X_test), y_test))

# One-hot encoding to "card_id", "zip_1"
def one_hot_encode(features):
    features["card_id"] = tf.one_hot(features["card_id"], depth=10)
    features["zip_1"] = tf.one_hot(features["zip_1"], depth=10)
    return features

train_dataset = train_dataset.map(lambda x, y: (one_hot_encode(x), y))
test_dataset = test_dataset.map(lambda x, y: (one_hot_encode(x), y))

# print for validate
for item in X_train.take(1):
    for key, value in item.items():
        print(f"{key}: {value.numpy()}")


# Count fraud or not
total_samples = len(y_train)
num_not_fraud = np.count_nonzero(y_train == 0)
num_fraud = np.count_nonzero(y_train == 1)

class_weight = {
    0: total_samples / (2 * num_not_fraud),
    1: total_samples / (2 * num_fraud)
}


# Build Neural Network
class Logistic_Model(tf.keras.Model):
    def __init__(self, units: int, kernel_l2_lambda: float, activity_l2_lambda: float, dropout_rate: float , kernel_initializer: str):
        super(Logistic_Model, self).__init__()

        self.units = units
        self.kernel_l2_lambda = kernel_l2_lambda
        self.activity_l2_lambda = activity_l2_lambda
        self.dropout_rate = dropout_rate
        self.kernel_initializer = kernel_initializer

        self.hidden = tf.keras.layers.Dense(
            units=self.units,
            kernel_regularizer=tf.keras.regularizers.L2(self.kernel_l2_lambda),
            activity_regularizer=tf.keras.regularizers.L2(self.activity_l2_lambda),
            activation="relu",
            kernel_initializer=self.kernel_initializer, # he_normal or he_uniform
            name="hidden"
        )

        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)

        self.output_layer = tf.keras.layers.Dense(1, activation="sigmoid")

    def call(self, inputs: tf.data.Dataset):
        card_id, amount, zip_1, zip_2, zip_3, user_avg_amount, merchant_avg_amount, \
        merchant_id, mcc, merchant_city, merchant_state, errors, use_chip, user_id = inputs

        x = tf.concat([card_id, amount, zip_1, zip_2, zip_3, user_avg_amount, merchant_avg_amount, 
                       merchant_id, mcc, merchant_city, merchant_state, errors, use_chip, user_id], axis=-1)
        
        x = self.hidden(x)
        x = self.dropout(x)
        output = self.output_layer(x)
        
        return output
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'units': self.units,
            'kernel_l2_lambda': self.kernel_l2_lambda,
            'activity_l2_lambda': self.activity_l2_lambda,
            'dropout_rate': self.dropout_rate,
            'kernel_initializer': self.kernel_initializer
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
    

# Define Objective function for Optuna
def Objective(trial):
    param = {
        "units": trial.suggest_int("units", 32, 2500),
        "kernel_l2_lambda": trial.suggest_float("kernel_l2_lambda", 1e-4, 1, log=True),
        "activity_l2_lambda": trial.suggest_float("activity_l2_lambda", 1e-4, 1, log=True),
        "dropout_rate": trial.suggest_float("kernel_l2_lambda", 0.0, 1, step=0.05),
        "kernel_initializer": trial.suggest_categorical("kernel_initializer", ["he_normal", "he_uniform"]),
        "lr": trial.suggest_float("activity_l2_lambda", 1e-4, 1, log=True),
        "batch_size": trial.suggest_int("batch_size", 16, 1024, step=8)
    }

    with open("nndb/nn_Hyper_1.txt", 'a') as f:
        f.write(str(param) + '\n')
    
    lr = param.pop("lr")
    batch_size = param.pop("batch_size")

    # try 3 times
    all_scores = []
    for _ in range(3):
        # Build CatBoost Classifier and Training
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
        model = Logistic_Model(**param)
        model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=lr), loss='binary_crossentropy', metrics=[tf.keras.metrics.F1Score(num_classes=1, threshold=0.5)])
        model.fit(train_dataset.batch(batch_size),
              epochs=1000,
              class_weight=class_weight,
              validation_data=test_dataset.batch(batch_size),
              callbacks=[early_stopping])

        # Predict & Validate
        y_pred = model.predict(X_test)
        model_metric = f1_score(y_test, y_pred)

        # Append Metric
        all_scores.append(model_metric)

    # Note Metric
    with open("nndb/nn_Hyper_1.txt", 'a') as f:
        f.write(f"F1 Score: {np.mean(all_scores)} \n\n")

    return np.mean(all_scores)


# Create Optuna sampler and study object
sampler = optuna.samplers.TPESampler(n_startup_trials=30)
study = optuna.create_study(sampler=sampler, 
    study_name="catboost_for_card_fraud_11", 
    direction="maximize", 
    storage="sqlite:///nndb/11.db", 
    load_if_exists=True)
study.optimize(Objective, n_trials=330, n_jobs=1)

# Print best hyper-parameter set
with open("nndb/nn_Hyper_1.txt",'a') as f:
    f.write(f"Best Hyper-parameter set: \n{study.best_params}\n")
    f.write(f"Best value: {study.best_value}")

print(f"Best Hyper-parameter set: \n{study.best_params}\n")
print(f"Best value: {study.best_value}")