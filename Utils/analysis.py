#%%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

path = r"C:\Users\yacou\Desktop\GP\Code\Graduation-Project\Data"
sns.set_style('darkgrid') # darkgrid, white grid, dark, white and ticks
plt.rc('axes', titlesize=18)     # fontsize of the axes title
plt.rc('axes', labelsize=14)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=13)    # fontsize of the tick labels
plt.rc('ytick', labelsize=13)    # fontsize of the tick labels
plt.rc('legend', fontsize=13)    # legend fontsize
plt.rc('font', size=13)          # controls default text sizes

color = sns.color_palette('deep')

plt.figure(figsize=(8,4), tight_layout=True)
# %%
df = pd.read_csv(path+"\Dataset_Full.csv")
df = df.iloc[::4,:]
barplot = df.groupby(['Status'], as_index=False).mean()
# %%
plt.bar(barplot['Status'], barplot['Age'], color=color)
plt.xlabel('Status')
plt.ylabel('Age')
plt.title('Average age of patients')
plt.show()

# %%
fig, ax = plt.subplots(nrows=1,ncols=3, figsize=(12, 5), tight_layout=True, sharey=True)

df_cancer = df[df['Status'].isin(['Cancer'])]
df_normal = df[df['Status'].isin(['Normal'])]
df_benign = df[df['Status'].isin(['Benign'])]
bins=[i for i in range(30,95,5)]

sns.histplot(ax=ax[0], data=df_cancer, x='Age',bins=bins, color=color[1], linewidth=2)
ax[0].set_title("Cancer")
sns.histplot(ax=ax[1], data=df_normal, x='Age',bins=bins, color=color[2], linewidth=2)
ax[1].set_title("Normal")
sns.histplot(ax=ax[2], data=df_benign, x='Age',bins=bins, color=color[3], linewidth=2)
ax[2].set_title("Benign")
