#%%
import pandas as pd

path = r"home/abulubad/Graduation-Project/Data/Data-MoreThanTwoMasks.xlsx"

sheet = pd.read_excel(path)

for i in range(len(sheet)):
    sheet.loc[i, ('fullPath')] = '\\' + sheet['fullPath'][i]
    sheet.loc[i, ('fileName')] = '\\' + sheet['fileName'][i]

# %%
sheet.to_csv('Dataset_Full.csv')
# %%
