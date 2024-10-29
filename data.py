
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



df_train = pd.read_csv("train.csv")



df_train.head()

# %%
num_sample = 4
df_sample  = df_train.sample(num_sample)
df_x_sample = df_sample.drop('label', axis=1)
df_y_sample = df_sample['label']

print(df_x_sample.head())
print(df_y_sample.head())


channels_name = ["TP9","FP1","FP2", "TP10"]
fig = plt.figure(figsize=(20, 20))  


channels = 4
fig.suptitle("EEG Signal")
for i in range(num_sample):
    for channel in range(channels):
        ax = fig.add_subplot(num_sample, channels, i * channels + channel + 1)
        ax.plot(df_x_sample.iloc[i, channel * 440: (channel + 1) * 440])
        ax.set_title(f"Label: {df_y_sample.iloc[i]} - Channel: {channels_name[channel]}")
        ax.set_xlabel("Time")
        ax.set_ylabel("Signal")

plt.savefig("eeg_signal.png")

