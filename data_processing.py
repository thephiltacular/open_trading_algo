#! /usr/bin/env python3

import pandas as pd

data_path = "23-03-07-TV-Data Input.csv"
df = pd.read_csv(data_path,header=0)
print(df.head())
# print(df.header_items())
print(df["Ticker"].head())