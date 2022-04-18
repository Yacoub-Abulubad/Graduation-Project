import pandas as pd

path = r"/home/abulubad/Graduation-Project/Data/Data-MoreThanTwoMasks.xlsx"

sheet = pd.read_excel(path)

for i in range(len(sheet)):
    sheet.loc[i, ('fullPath')] = '/' + sheet['fullPath'][i].replace("\\", "/")
    sheet.loc[i, ('fileName')] = '/' + sheet['fileName'][i].replace("\\", "/")

#sheet.replace(r'^\\.$', '/')
sheet.to_csv('/home/abulubad/Graduation-Project/Data/Dataset_Full.csv')
