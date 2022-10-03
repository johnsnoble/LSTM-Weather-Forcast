import tensorflow as tf
import pandas as pd
from tensorflow import keras
from datetime import datetime
import numpy as np
from tensorflow import keras

df = pd.read_csv("data/era5.csv")
attr = ["temperature_2m (°C)","relativehumidity_2m (%)","dewpoint_2a (°C)","surface_pressure (hPa)","precipitation (mm)","cloudcover (%)","direct_normal_irradiance (W/m²)","windspeed_100m (km/h)","vapor_pressure_deficit (kPa)"]

def elem(x,xs):
    for i in xs:
        if x==i:
            return True
    return False
  
for i in df: 
    if not elem(i,attr):
        del df[i]

df.fillna(method='pad',inplace=True)

split = (0.7,0.15)
past = 6

def getTraining(df,past):
  np_df = df.to_numpy();
  x = []
  y = []
  for i in range(len(np_df)-past):
    row = [r for r in np_df[i:i+past]]
    x.append(row)
    y.append([np_df[i+past][0],np_df[i+past][3]])
  return np.array(x),np.array(y)

x_data,y_data = getTraining(df,past)

def normalise(x,n,m,s):
  if s == 0:
    s = 1
  x[:,:,n] = (x[:,:,n] - m)/s

def normalise_out(y,n,m,s):
  if s == 0:
    s = 1
  y[:,n] = (y[:,n]-m)/s

means = []
stds = []
for i in range(7):
  means.append(np.mean(x_data[:,:,i]))
  stds.append(np.std(x_data[:,:,i]))
  normalise(x_data,i,means[i],stds[i])

means_out = []
stds_out = []
for i in range(2):
  means_out.append(np.mean(y_data[:,i]))
  stds_out.append(np.std(y_data[:,i]))
  normalise_out(y_data,i,means_out[i],stds_out[i])

train_split = int(x_data.shape[0]*split[0])
val_split = train_split + int(x_data.shape[0]*split[1])
x_train = x_data[:train_split]
y_train = y_data[:train_split]
x_val = x_data[train_split:val_split]
y_val = y_data[train_split:val_split]
x_test = x_data[val_split:]
y_test = y_data[val_split:]
x_train.shape, y_train.shape,x_val.shape,y_val.shape,x_test.shape,y_test.shape

model1 = keras.models.Sequential()
model1.add(keras.layers.InputLayer((past,8)))
model1.add(keras.layers.LSTM(64))
model1.add(keras.layers.Dense(8,'relu'))
model1.add(keras.layers.Dense(2,'linear'))

model1.summary()

cb = keras.callbacks.ModelCheckpoint('model1/',monitor="val_loss",save_best_only=True)
model1.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001),loss='mse')

model1.fit(x_train,y_train,validation_data=(x_val,y_val),epochs=10,callbacks=[cb])

print("model trained!")
