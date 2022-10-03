from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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

def plot_predictions(model,x,y,start=0,end=500000):
  predictions = model.predict(x[start:end])
  for i in range(2):
    plt.plot(predictions[:,i])
    plt.plot(y[start:end][:,i])
  plt.show()

model = keras.models.load_model("model1/")
plot_predictions(model,x_data,y_data,390200,390400)
