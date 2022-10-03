import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("data/era5.csv")
plt.matshow(df.corr())
plt.xticks(range(df.shape[1]), df.columns, rotation=90)
plt.gca().xaxis.tick_bottom()
plt.yticks(range(df.shape[1]), df.columns)
plt.show()
