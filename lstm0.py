import tensorflow as tf
import pandas as pd
from tensorflow import keras
from datetime import datetime
import numpy as np
from tensorflow import keras

df = pd.read_csv("data/era5.csv")
attr = ["temperature_2m (°C)","relativehumidity_2m (%)","dewpoint_2a (°C)","surface_pressure (hPa)","precipitation (mm)","cloudcover (%)","direct_normal_irradiance (W/m²)","windspeed_100m (km/h)","winddirection_100m (°)","vapor_pressure_deficit (kPa)"]

temp = df["temperature_2m (°C)"]

split = (0.7,0.15)
past = 6

def getTraining(df,past):
  np_df = df.to_numpy()
  x = []
  y = []
  for i in range(len(np_df)-past):
    row = [[a] for a in np_df[i:i+past]]
    x.append(row)
    label = np_df[i+past]
    y.append(label)
  return np.array(x),np.array(y)

x_data,y_data = getTraining(temp,window_size)

train_split = int(x_data.shape[0]*split[0])
val_split = train_split + int(x_data.shape[0]*split[1])
x_train = x_data[:train_split]
y_train = y_data[:train_split]
x_val = x_data[train_split:val_split]
y_val = y_data[train_split:val_split]
x_test = x_data[val_split:]
y_test = y_data[val_split:]

model = keras.models.Sequential()
model.add(keras.layers.InputLayer((6,1)))
model.add(keras.layers.LSTM(64))
model.add(keras.layers.Dense(8,'relu'))
model.add(keras.layers.Dense(1,'linear'))

model.summary()

cb = keras.callbacks.ModelCheckpoint('model0/',monitor="val_loss",save_best_only=True)
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001),loss='mse')

model.fit(x_train,y_train,validation_data=(x_val,y_val),epochs=10,callbacks=[cb])

model = keras.models.load_model('model0/')

predictions = model.predict(x_train).flatten()
result = pd.DataFrame(data={"Predictions":predictions,"Actual":y_train})
