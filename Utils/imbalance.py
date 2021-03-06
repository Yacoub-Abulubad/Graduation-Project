#%%
from Dataset import DataSequenceLoader

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

path = "/home/abulubad/Graduation-Project/Data"
train_data = DataSequenceLoader(path, 1)
val_data = DataSequenceLoader(path, 1, is_val=True)

# %%
sns.set_style('darkgrid')  # darkgrid, white grid, dark, white and ticks
plt.rc('axes', titlesize=18)  # fontsize of the axes title
plt.rc('axes', labelsize=14)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=13)  # fontsize of the tick labels
plt.rc('ytick', labelsize=13)  # fontsize of the tick labels
plt.rc('legend', fontsize=13)  # legend fontsize
plt.rc('font', size=13)  # controls default text sizes

color = sns.color_palette('deep')

plt.figure(figsize=(8, 4), tight_layout=True)
# %%
df = pd.DataFrame(train_data.y_path, columns=["Status"])
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
df["dummy"] = range(1, len(df) + 1)
barplot = df.groupby(['Status'], as_index=False).count()
# %%
plt.bar(barplot['Status'], barplot['dummy'], color=color)
plt.xlabel('Status')
plt.ylabel('Count')
plt.title('Training data')
plt.show()
# %%
val_df = pd.DataFrame(val_data.y_path, columns=["Status"])
val_df = val_df.loc[:, ~val_df.columns.str.contains('^Unnamed')]
val_df["dummy"] = range(1, len(val_df) + 1)
barplot = val_df.groupby(['Status'], as_index=False).count()
# %%
plt.bar(barplot['Status'], barplot['dummy'], color=color)
plt.xlabel('Status')
plt.ylabel('Count')
plt.title('Validation data')
plt.show()

# %%
