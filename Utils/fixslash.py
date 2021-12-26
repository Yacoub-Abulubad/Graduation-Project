#%%
import pandas as pd
path = r"C:\Users\yacou\Desktop\Studies\1-Deep Learning\GP\Code\Dataset\JPEG-Trial"

sheet = pd.read_excel(path + '\Dataset.xlsx')

for i in range(len(sheet)):
    sheet.loc[i,('fullPath')] = '\\' + sheet['fullPath'][i]
    sheet.loc[i,('fileName')] = '\\' + sheet['fileName'][i]

    
# %%
sheet.to_csv('Dataset/JPEG-Trial/Dataset.csv')
# %%
