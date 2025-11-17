import os
import pandas as pd
import numpy as np
import qrcode
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

dp1 = processed/*.csv
dp2 = processed/*.csv

row_index_a = 0
row_index_b = 0
for_qr = True
EPOCHS = 20
BATCH_SIZE = 32
RANDOM_STATE = 42
#numeric columns only
def load_numeric_df(path):
    df = pd.read_csv(path)
    df_num = df.select_dtypes(include=[np.number]).copy()
    return df, df_num
#droping unnecessary info
def fix_dataset2(df_num):
    drop_cols = [
        "median_sale_price", "median_list_price", "median_ppsf",
        "median_list_ppsf", "price", "date", "city", "city_full"
    ]
    for col in drop_cols:
        if col in df_num.columns:
            df_num = df_num.drop(columns=[col])
    return df_num.select_dtypes(include=[np.number]).copy()
#small dense autoencoder on each dataset
def build_dense_autoencoder(input_len, latent_dim=16):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_len,)),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(latent_dim, activation="relu"),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(input_len, activation=None)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model
def train_autoencoder(X):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    X_train, X_val = train_test_split(Xs, test_size=0.2, random_state=RANDOM_STATE)
    model = build_dense_autoencoder(input_len=Xs.shape[1], latent_dim=min(32, Xs.shape[1]))
    model.fit(
        X_train, X_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, X_val),
        verbose=2
    )
    return model, scaler
def reconstruct(model, scaler, row):
    r_scaled = scaler.transform(row.reshape(1, -1))
    r_rec_scaled = model.predict(r_scaled)
    return scaler.inverse_transform(r_rec_scaled)[0]
def vector_to_text(title, vec, columns):
    lines = [f""]
    for col, val in zip(columns, vec):
        if abs(val - np.round(val)) < 1e-6:
            sval = str(int(np.round(val)))
        else:
            sval = f"{float(np.round(val,4))}"
        lines.append(f"{col}: {sval}")
    return "\n".join(lines) + "\n\n"

print("loading 1 dataset")
dfA_full, dfA = load_numeric_df(dp1)
modelA, scalerA = train_autoencoder(dfA.values)
origA = dfA.values[row_index_a]
outA = reconstruct(modelA, scalerA, origA) if for_qr else origA
print("loading 2 dataset")
dfB_full, dfB = load_numeric_df(dp2)
dfB = fix_dataset2(dfB)
modelB, scalerB = train_autoencoder(dfB.values)
origB = dfB.values[row_index_b]
outB = reconstruct(modelB, scalerB, origB) if for_qr else origB
textA = vector_to_text("DATASET 1", outA, dfA.columns)
textB = vector_to_text("DATASET 2", outB, dfB.columns)
final_text = textA + textB
qr_path = "qr_combined1.png"
qr = qrcode.make(final_text)
qr.save(qr_path)

print("\nSaved ONE QR code:", qr_path)

