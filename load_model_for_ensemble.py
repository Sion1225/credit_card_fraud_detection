import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
import catboost
import lightgbm as lgb
import tensorflow as tf

# Read "train.csv" file
df = pd.read_csv("DataSet/test.csv")

# Splitting the 'is_fraud?' column
#labels = df["is_fraud?"].copy().to_numpy()
#labels = labels.astype(int)
#df = df.drop("is_fraud?", axis=1)

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
#print(labels[:5])
print(df.head(5))

# Validate
print(df.dtypes)

# Split Datas for train & test
#X_train, X_test, y_train, y_test = train_test_split(df, labels, test_size=0.1, random_state=1225)

# Drop index
output = pd.DataFrame(index=df["index"]) # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
#X_train = X_train.set_index(X_train.columns[0])
#X_test = X_test.set_index(X_test.columns[0])
df = df.set_index(df.columns[0])

# Catboost
def catboost_function(X_test): #, y_test
    print("===========CatBoost===========")
    # Load model
    model = catboost.CatBoostClassifier()
    model.load_model("catboostdb/catboost_full_data_model_4_0.6143405134257893_20230829-020535.cbm")

    # Test loaded model
    #y_pred = model.predict(X_test)

    # Validation model 
    #print(f1_score(y_pred, y_test))

    # Predict Probability & save to dataframe
    y_pred = model.predict(X_test, prediction_type='Probability')

    print("===============================================")

    return y_pred


# Create additional data
# User average amount
def add_data(df):
    print("Making additional datas")
    user_avg_amount = df.groupby("user_id")["amount"].mean().reset_index()
    user_avg_amount.columns = ['user_id', 'user_avg_amount']

    # Merchant average amount
    merchant_avg_amount = df.groupby("merchant_id")["amount"].mean().reset_index()
    merchant_avg_amount.columns = ["merchant_id", "merchant_avg_amount"]

    # Add to Original Dataset
    df = pd.merge(df, user_avg_amount, on="user_id", how="left")
    df = pd.merge(df, merchant_avg_amount, on="merchant_id", how="left")

    return df


# LightGBM
def lightgbm_fucntion(X_test): #, y_test
    print("===========LightGBM===========")
    model = lgb.Booster(model_file='lightgbmdb/lightgbm_full_data_model_1_0.565894417014812_20230904-041116.txt')

    # Test loaded model
    #y_pred = np.round(model.predict(X_test))

    # Validation model 
    #print(f1_score(y_pred, y_test))

    # Predict Probability & save to dataframe
    y_pred = model.predict(X_test)

    print("===============================================")

    return y_pred


# Pre-processing for NN
def preprocessing_NN():
    # Read "train.csv" file
    df = pd.read_csv("DataSet/test.csv")

    # Splitting the 'is_fraud?' column
    #labels = df["is_fraud?"].copy().to_numpy()
    #labels = labels.astype(int)
    #df = df.drop("is_fraud?", axis=1)
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

    # errors? -> errors
    df.rename(columns={"errors?" : "errors"}, inplace=True)

    # Print Data sample
    print(f"\n{df.head(5)}\n")
    #print(labels[:5])
    print()
    print(le_errors.classes_)
    print()

    # Validate
    print(df.dtypes)

    # Split Datas for train & test
    #X_train, X_test, y_train, y_test = train_test_split(df, labels, test_size=0.1, random_state=1225)

    # Shift to tf.data.Dataset
    #train_dataset = tf.data.Dataset.from_tensor_slices((dict(X_train.to_dict('list')), y_train))
    #test_dataset = tf.data.Dataset.from_tensor_slices((dict(X_test.to_dict("list")), y_test))
    df_dataset = tf.data.Dataset.from_tensor_slices((dict(df.to_dict("list")))) #, labels

    # One-hot encoding to "card_id", "zip_1"
    def one_hot_encode(features):
        features["card_id"] = tf.one_hot(features["card_id"], depth=10)
        features["zip_1"] = tf.one_hot(features["zip_1"], depth=10)
        features["use_chip"] = tf.one_hot(features["use_chip"], depth=3)
        return features

    #train_dataset = train_dataset.map(lambda x, y: (one_hot_encode(x), y))
    #test_dataset = test_dataset.map(lambda x, y: (one_hot_encode(x), y))
    df_dataset = df_dataset.map(lambda x : (one_hot_encode(x)))

    # Change to vector
    def reshape_scalars(x): #, y
        reshaped_x = {}
        for key, value in x.items():
            if len(value.shape) == 0:  # 스칼라 값인 경우
                reshaped_x[key] = tf.cast(tf.reshape(value, (1,)), dtype=tf.float32)
            else:
                reshaped_x[key] = tf.cast(value, dtype=tf.float32)
        return reshaped_x #, y

    # Dataset 객체에 map 함수 적용
    #train_dataset = train_dataset.map(reshape_scalars)
    #test_dataset = test_dataset.map(reshape_scalars)
    df_dataset = df_dataset.map(reshape_scalars)

    # print for validate
    for item in df_dataset.take(1): #, label
        for key, value in item.items():
            print(f"{key}: {value.numpy()}")
        #print(label)

    #return train_dataset, test_dataset, df_dataset
    return df_dataset


# FFNN
def NN_fucntion(X_test): #, y_test
    print("===========FFNN===========")
    model = tf.keras.models.load_model('nndb/nn_model_1')
    batch_size = 107
    X_test = X_test.batch(batch_size)

    # Test loaded model
    #y_pred = model.predict(X_test)
    #y_pred = (y_pred > 0.5).astype(int).flatten()

    # Validation model 
    #print(f1_score(y_pred, y_test))

    # Predict Probability & save to dataframe
    y_pred = model.predict(X_test)

    print("===============================================")

    return y_pred


# Catboost
y_pred = catboost_function(df)
output["catboost"] = y_pred[:,1]

# Add data
df = add_data(df)
print(df.head(5))
# Split Datas for train & test again
#X_train, X_test, y_train, y_test = train_test_split(df, labels, test_size=0.1, random_state=1225)
# Drop index
#X_train = X_train.set_index(X_train.columns[0])
#X_test = X_test.set_index(X_test.columns[0])

# LightGBM
y_pred = lightgbm_fucntion(df)
output["lightgbm"] = y_pred

# Pre-processing for FFNN
df_dataset = preprocessing_NN()
print(df_dataset)
y_pred = NN_fucntion(df_dataset)
y_pred = y_pred.flatten()
output["NN"] = y_pred

# Print
print(output.head(50))
output.to_csv("ensembledb/test_2.csv")