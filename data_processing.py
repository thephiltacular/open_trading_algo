#! /usr/bin/env python3

import pandas as pd

data_path = "23-03-07-TV-Data Input.csv"

# store each day in a df indexed by index and date
# create automated read in function that adds day to df, then outputs unified csv
# base 
# relative volume per n days in the past
# d-relative-volumn -> volume relative to the last n days
# compare volume of day to relative volume over n days

#  4 categories of cells,
# grey reading in of data
# green trend calculations based on various algorithms
# orange 
# PR is percent rank of indicator in relation to other tickers
# example: the ticker with most volume gets 100%, the least gets 0%
# generate a set of parameters based on model and store in time series
# create regression model to optimize parameters for each day
df = pd.read_csv(data_path,header=0)
# print(df.head())
# print(df.header_items())
# print(df["Ticker"].head())


data_folder = "data/"
data_dict = {"data": {},
             "alerts": {}}
# data = pd.DataFrame()

with open("raw_data.txt") as files:
    for file in files:
        file = file.replace("\n", "")
        df_temp = pd.read_csv(data_folder + file, header=0) 
        if "A" in file:  # Alert file
            name = file.replace("A.csv", "")
            data_dict["alerts"][name] = df_temp
            print(name)
        else:  # Data file
            name = file.replace("D.csv", "")
            data_dict["data"][name] = df_temp
            print(name)
for name, day in data_dict.items():
    print("==========================================")
    print(name)
    for label, data in day.items():
        print(data.head())
print("==========================================\n\n\n\n\n")
print(data_dict)

            
            
        