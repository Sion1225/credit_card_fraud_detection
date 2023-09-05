import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
import optuna

import f_score_metrics

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

# Replace NaN 
df["merchant_state"] = df["merchant_state"].fillna("Online")
df = df.fillna(-1)

# Change float64 with int64
df["amount"] = round(df["amount"] * 10)
df["amount"] = df["amount"].astype("int64")
df["zip_2"] = df["zip_2"].astype("int64")
df["zip_4"] = df["zip_4"].astype("int64")

df["merchant_id"] = df["merchant_id"].astype("int64")
df["mcc"] = df["mcc"].astype("int64")
df["merchant_city"] = df["merchant_city"].astype("category")
df["merchant_state"] = df["merchant_state"].astype("category")
df["errors?"] = df["errors?"].astype("category")
df["use_chip"] = df["use_chip"].astype("category")
df["user_id"] = df["user_id"].astype("int64")
df["card_id"] = df["card_id"].astype("int64")
df["zip_1"] = df["zip_1"].astype("int64")

print(df.head(5))

# Create additional data
# User average amount
user_avg_amount = df.groupby("user_id")["amount"].mean().reset_index()
user_avg_amount['amount'] = np.round(user_avg_amount['amount'])
user_avg_amount.columns = ['user_id', 'user_avg_amount']

# Merchant average amount
merchant_avg_amount = df.groupby("merchant_id")["amount"].mean().reset_index()
merchant_avg_amount['amount'] = np.round(merchant_avg_amount['amount'])
merchant_avg_amount.columns = ["merchant_id", "merchant_avg_amount"]

# Add to Original Dataset
df = pd.merge(df, user_avg_amount, on="user_id", how="left")
df = pd.merge(df, merchant_avg_amount, on="merchant_id", how="left")


# Labeling to categorical datas
le_city = LabelEncoder()
le_city.fit(df["merchant_city"])
df["merchant_city"] = le_city.transform(df["merchant_city"])

le_state = LabelEncoder()
le_state.fit(df["merchant_state"])
df["merchant_state"] = le_state.transform(df["merchant_state"])

le_errors = LabelEncoder()
le_errors.fit(df["errors?"])
df["errors?"] = le_errors.transform(df["errors?"])

le_chip = LabelEncoder()
le_chip.fit(df["use_chip"])
df["use_chip"] = le_chip.transform(df["use_chip"])


# Print Data sample
print(f"\n{df.head(5)}\n")
print(labels[:5])
print()
print(le_errors.classes_)
print()

# Validate
print(df.dtypes)

# Split Datas for train & test
X_train, X_test, y_train, y_test = train_test_split(df, labels, test_size=0.1, random_state=1225)

# Shift to tf.data.Dataset
train_dataset = tf.data.Dataset.from_tensor_slices((dict(X_train.to_dict('list')), y_train))
test_dataset = tf.data.Dataset.from_tensor_slices((dict(X_test.to_dict("list")), y_test))

# One-hot encoding to "card_id", "zip_1"
def one_hot_encode(features):
    features["card_id"] = tf.one_hot(features["card_id"], depth=10)
    features["zip_1"] = tf.one_hot(features["zip_1"], depth=10)
    features["use_chip"] = tf.one_hot(features["use_chip"], depth=3)
    return features

train_dataset = train_dataset.map(lambda x, y: (one_hot_encode(x), y))
test_dataset = test_dataset.map(lambda x, y: (one_hot_encode(x), y))

# Change to vector
def reshape_scalars(x, y):
    reshaped_x = {}
    for key, value in x.items():
        if len(value.shape) == 0:  # 스칼라 값인 경우
            reshaped_x[key] = tf.cast(tf.reshape(value, (1,)), dtype=tf.float32)
        else:
            reshaped_x[key] = tf.cast(value, dtype=tf.float32)
    return reshaped_x, y

# Dataset 객체에 map 함수 적용
train_dataset = train_dataset.map(reshape_scalars)
test_dataset = test_dataset.map(reshape_scalars)

# print for validate
for item, label in train_dataset.take(1):
    for key, value in item.items():
        print(f"{key}: {value.numpy()}")
    print(label)

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
    def __init__(self, units: int, output_dim:int, kernel_l2_lambda: float, activity_l2_lambda: float, dropout_rate: float , kernel_initializer: str):
        super(Logistic_Model, self).__init__()

        self.units = units
        self.output_dim = output_dim
        self.kernel_l2_lambda = kernel_l2_lambda
        self.activity_l2_lambda = activity_l2_lambda
        self.dropout_rate = dropout_rate
        self.kernel_initializer = kernel_initializer

        self.input_user_id = tf.keras.layers.Embedding(input_dim=2000, output_dim=self.output_dim, input_length=1, mask_zero=False)
        self.input_amount = tf.keras.layers.Embedding(input_dim=20000, output_dim=self.output_dim, input_length=1, mask_zero=False)
        self.input_mer_id = tf.keras.layers.Embedding(input_dim=25076, output_dim=self.output_dim, input_length=1, mask_zero=False)
        self.input_mer_ct = tf.keras.layers.Embedding(input_dim=4400, output_dim=self.output_dim, input_length=1, mask_zero=False)
        self.input_mer_st = tf.keras.layers.Embedding(input_dim=130, output_dim=self.output_dim, input_length=1, mask_zero=False)
        self.input_mcc = tf.keras.layers.Embedding(input_dim=110, output_dim=self.output_dim, input_length=1, mask_zero=False)
        self.input_zip2 = tf.keras.layers.Embedding(input_dim=1000, output_dim=self.output_dim, input_length=1, mask_zero=False)
        self.input_zip4 = tf.keras.layers.Embedding(input_dim=100, output_dim=self.output_dim, input_length=1, mask_zero=False)
        self.input_user_avg = tf.keras.layers.Embedding(input_dim=2000, output_dim=self.output_dim, input_length=1, mask_zero=False)
        self.input_mer_avg = tf.keras.layers.Embedding(input_dim=2000, output_dim=self.output_dim, input_length=1, mask_zero=False)

        self.input_card_id = tf.keras.layers.Dense(units=output_dim, activation="relu", kernel_initializer="he_normal")
        self.input_use_chip = tf.keras.layers.Dense(units=output_dim, activation="relu", kernel_initializer="he_normal")
        self.input_zip1 = tf.keras.layers.Dense(units=output_dim, activation="relu", kernel_initializer="he_normal")

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

        user_id_out = self.input_user_id(inputs["user_id"])
        user_id_out = tf.squeeze(user_id_out, axis=1)
        print(tf.shape(user_id_out))
        amount_out = self.input_amount(inputs["amount"])
        amount_out = tf.squeeze(amount_out, axis=1)
        print(tf.shape(amount_out))
        mer_id_out = self.input_mer_id(inputs["merchant_id"])
        mer_id_out = tf.squeeze(mer_id_out, axis=1)
        print(tf.shape(mer_id_out))
        mer_ct_out = self.input_mer_ct(inputs["merchant_city"])
        mer_ct_out = tf.squeeze(mer_ct_out, axis=1)
        print(tf.shape(mer_ct_out))
        mer_st_out = self.input_mer_st(inputs["merchant_state"])
        mer_st_out = tf.squeeze(mer_st_out, axis=1)
        print(tf.shape(mer_st_out))
        mcc_out = self.input_mcc(inputs["mcc"])
        mcc_out = tf.squeeze(mcc_out, axis=1)
        print(tf.shape(mcc_out))
        zip2_out = self.input_zip2(inputs["zip_2"])
        zip2_out = tf.squeeze(zip2_out, axis=1)
        print(tf.shape(zip2_out))
        zip4_out = self.input_zip4(inputs["zip_4"])
        zip4_out = tf.squeeze(zip4_out, axis=1)
        print(tf.shape(zip4_out))
        user_avg_out = self.input_user_avg(inputs["user_avg_amount"])
        user_avg_out = tf.squeeze(user_avg_out, axis=1)
        print(tf.shape(user_avg_out))
        mer_avg_out = self.input_mer_avg(inputs["merchant_avg_amount"])
        mer_avg_out = tf.squeeze(mer_avg_out, axis=1)
        print(tf.shape(mer_avg_out))

        card_id_out = self.input_card_id(inputs["card_id"])
        print(tf.shape(card_id_out))
        use_chip_out = self.input_use_chip(inputs["use_chip"])
        print(tf.shape(use_chip_out))
        zip1_out = self.input_zip1(inputs["zip_1"])
        print(tf.shape(zip1_out))
        
        x = tf.concat([user_id_out, card_id_out, amount_out, inputs["errors?"], mer_id_out, mer_ct_out, mer_st_out, 
                       mcc_out, mcc_out, use_chip_out, zip1_out, zip2_out, zip4_out, user_avg_out, mer_avg_out], axis=1)

        x = self.hidden(x)
        x = self.dropout(x)
        output = self.output_layer(x)
        
        return output
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'units': self.units,
            'output_dim': self.output_dim,
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
        "output_dim": trial.suggest_int("units", 1, 500),
        "kernel_l2_lambda": trial.suggest_float("kernel_l2_lambda", 1e-4, 1, log=True),
        "activity_l2_lambda": trial.suggest_float("activity_l2_lambda", 1e-4, 1, log=True),
        "dropout_rate": trial.suggest_float("dropout_rate", 0.0, 1, step=0.05),
        "kernel_initializer": trial.suggest_categorical("kernel_initializer", ["he_normal", "he_uniform"]),
        "lr": trial.suggest_float("activity_l2_lambda", 1e-4, 1, log=True),
        "batch_size": trial.suggest_int("batch_size", 16, 1024, step=8)
    }

    with open("nndb/nn_Hyper_1.txt", 'a') as f:
        f.write(str(param) + '\n')
    
    lr = param.pop("lr")
    batch_size = param.pop("batch_size")

    # try 2 times
    all_scores = []
    for _ in range(2):
        # Build CatBoost Classifier and Training
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
        model = Logistic_Model(**param)
        model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=lr), loss='binary_crossentropy', metrics=[f_score_metrics.F1Score()])
        model.fit(train_dataset.batch(batch_size),
              epochs=1000,
              class_weight=class_weight,
              validation_data=test_dataset.batch(batch_size),
              callbacks=[early_stopping])

        # Predict & Validate
        y_pred = model.predict(test_dataset.batch(batch_size))
        y_pred = (y_pred > 0.5).astype(int).flatten()  # Convert probabilities to binary labels and flatten to 1D array
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
