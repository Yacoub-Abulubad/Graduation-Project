#%%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

path = r"/home/abulubad/Graduation-Project/Data"
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
df = pd.read_csv(path + "/Dataset_Full.csv")
df = df.iloc[::4, :]
barplot = df.groupby(['Status'], as_index=False).mean()
# %%
plt.bar(barplot['Status'], barplot['Age'], color=color)
plt.xlabel('Status')
plt.ylabel('Age')
plt.title('Average age of patients')
plt.show()

# %%



# %%
