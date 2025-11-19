import os
import pandas as pd
import numpy as np
import qrcode
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

dp1 = r"C:/Users/user/Desktop/F20DL-CW/processed/usa_real_estate.csv"
dp2 = r"C:/Users/user/Desktop/F20DL-CW/processed/zipcodes.csv"

QR_SIZE = 64 #can be increased but it gets slower
MAX_SAMPLES = 10000
EPOCHS = 20
BATCH_SIZE = 32
RANDOM_STATE = 42
def numeric_load(path):
    df = pd.read_csv(path)
    dfnum = df.select_dtypes(include=[np.number]).copy()
    return df, dfnum
#dropping unnessesary information
def clear(dfnum, is_dataset2=False):
    drop_c = [
        "median_sale_price", "median_list_price", "median_ppsf",
        "median_list_ppsf", "date"
    ]
    for col in drop_c:
        if col in dfnum.columns:
            dfnum = dfnum.drop(columns=[col])
    if is_dataset2 and "price" in dfnum.columns:
        dfnum = dfnum.drop(columns=["price"])
    return dfnum.select_dtypes(include=[np.number]).copy()
#numeric features as text for our qrcode
def row_to_text(row, columns):
    lines = []
    for col, val in zip(columns, row):
        lines.append(f"{col}:{val:.4f}")
    return "\n".join(lines)
def make_qr(text):
    qr = qrcode.QRCode(box_size=1, border=1)
    qr.add_data(text)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    img = img.resize((QR_SIZE, QR_SIZE))
    return np.array(img, dtype=np.float32) / 255.0

#loading and clearing data
df1, df1num = numeric_load(dp1)
df2, df2num = numeric_load(dp2)
df1_clean = clear(df1num, is_dataset2=False)  
df2_clean = clear(df2num, is_dataset2=True)
#grouping
df2_grouped = df2_clean.groupby("zipcode").mean().reset_index()
#merging
merged = pd.merge(df1_clean, df2_grouped, how="left", left_on="zip_code", right_on="zipcode")
merged = merged.iloc[:MAX_SAMPLES].copy()
PRICE_COL = "price"  # from dataset1
merged_num = merged.select_dtypes(include=[np.number]).copy()
merged_num["price_target"] = merged_num[PRICE_COL]
#preparing features
X = merged_num.drop(columns=["price_target"])
y = merged_num["price_target"].values
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
#convert to qrcode
print("Generating QR codes...")
cols = X.columns.tolist()
rows = X_scaled
imgs = np.zeros((len(rows), QR_SIZE, QR_SIZE, 1), dtype=np.float32)
for i, row in enumerate(rows):
    text = row_to_text(row, cols)
    img = make_qr(text)
    imgs[i, :, :, 0] = img
print("QR generation complete.")
X_train, X_test, y_train, y_test = train_test_split(
    imgs, y, test_size=0.2, random_state=RANDOM_STATE
)
#cnn
cnn = tf.keras.Sequential([
    tf.keras.layers.Conv2D(16, (3,3), activation="relu", input_shape=(QR_SIZE, QR_SIZE,1)),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Conv2D(32, (3,3), activation="relu"),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(1) 
])
cnn.compile(optimizer="adam", loss="mse")
cnn.summary()
#train
history = cnn.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE
)

#plotting loss
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='val')
plt.xlabel('epoch')
plt.ylabel('MSE loss')
plt.legend()
plt.title("training and validation loss")
plt.show()
#predicted and actual price
y_pred = cnn.predict(X_test)
plt.scatter(y_test, y_pred, alpha=0.3)
plt.xlabel("AP")#actual
plt.ylabel("PP")#predicted
plt.title("CNN PP vs AP")
plt.show()
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"MSE: {mse:.2f}, MAE: {mae:.2f}, R²: {r2:.4f}")
