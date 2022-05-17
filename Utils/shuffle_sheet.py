#%%
import pandas as pd

path = r"/home/abulubad/Graduation-Project/Data/Dataset_Full.csv"

sheet = pd.read_csv(path)

sheet = sheet.sample(frac=1).reset_index(drop=True)
sheet = sheet.loc[:, ~sheet.columns.str.contains('^Unnamed')]
#%%
sheet.to_csv(
    '/home/abulubad/Graduation-Project/Data/Dataset_Full_Shuffled.csv')

# %%
