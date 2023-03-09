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


class Model():
    def __init__(self):
        data_folder = "data/"
        self.data_dict = {"data": {},
                    "alerts": {}}
        # data = pd.DataFrame()

        with open("raw_data.txt") as files:
            for file in files:
                file = file.replace("\n", "")
                df_temp = pd.read_csv(data_folder + file, header=0) 
                if "A" in file:  # Alert file
                    name = file.replace("A.csv", "")
                    self.data_dict["alerts"][name] = df_temp
                    print(name)
                else:  # Data file
                    name = file.replace("D.csv", "")
                    self.data_dict["data"][name] = df_temp
                    print(name)

        self.data = self.data_dict["data"]
        self.calculate_relative_volume_div_by()
        self.calculate_d_relative_volume_up_or_down()
        ###################################### End __init__
        
    def calculate_relative_volume_div_by(self):
            
            cols_out = ["/10",
                    "/30",
                    "/60",
                    "/90",
                    " 10/30"]
            cols = ["Average Volume (10 day)",
                    "Average Volume (30 day)",
                    "Average Volume (60 day)",
                    "Average Volume (90 day)"]
            for day, df in self.data.items():
                # print("=======================")
                # print(day)
                cols_to_print = ["Ticker"]
                # Calculate d-relative volume col
                # df["D-Relative Volume"] = df.apply(
                #         lambda row: self.d_relative_volume(row["Description"], row["Relative Volume"]),
                #         axis=1
                #     )
                # Calculate 
                for col_out, col in zip(cols_out, cols):
                    col_out_name = "Relative Volume" + col_out
                    df[col_out_name] = df.apply(
                        lambda row: self.relative_volume_div(row["Volume"], row[col], row["Description"]),
                        axis=1)
                    # percentile_rank_col = col_out_name + "(pct rank)"
                    # print("col:", col, "col_out:", col_out_name, "pct_rank:", percentile_rank_col)
                    # df[percentile_rank_col] = df[col_out_name].rank(pct=True)
                    cols_to_print += [col, 
                                      col_out_name, 
                                    #   percentile_rank_col
                                      ]
                    # print(df[percentile_rank_col].head())
                
                # print(cols_to_print)
                # print(df[cols_to_print])
                

    def calculate_d_relative_volume_up_or_down(self):
            
            cols_out = [
                "D-Relative Volume-Up",
                "Relative Volume/10-Up",
                "Relative Volume/30-Up",
                "Relative Volume/60-Up",
                "Relative Volume/90-Up",
                "D-Relative Volume-Down",
                "Relative Volume/10-Down",
                "Relative Volume/30-Down",
                "Relative Volume/60-Down",
                "Relative Volume/90-Down"
            ]
            cols = [
                "D-Relative Volume",
                "Average Volume (10 day)",
                "Average Volume (30 day)",
                "Average Volume (60 day)",
                "Average Volume (90 day)",
                "D-Relative Volume",   
                "Average Volume (10 day)",
                "Average Volume (30 day)",
                "Average Volume (60 day)",
                "Average Volume (90 day)"]
            for day, df in self.data.items():
                print("=======================")
                print(day)
                cols_to_print = ["Ticker"]
                # Calculate d-relative volume col
                df["D-Relative Volume"] = df.apply(
                        lambda row: self.d_relative_volume(row["Relative Volume"], row["Description"]),
                        axis=1
                    )
                # Calculate 
                for col_out, col in zip(cols_out, cols):
                    # col_out_name = "Relative Volume" + col_out
                    # up = True 
                    if "Up" in col_out:
                        # up = False
                        df[col_out] = df.apply(
                            lambda row: self.d_relative_volume_up(row[col], row["Description"]),
                            axis=1)
                    else:
                        df[col_out] = df.apply(
                            lambda row: self.d_relative_volume_down(row[col], row["Description"]),
                            axis=1)
                    cols_to_print += [col, col_out]
                    # print(df[percentile_rank_col].head())
                
                print(cols_to_print)
                print(df[cols_to_print])

       
    def d_relative_volume(self, relative_volume, description):
        return relative_volume - 1.0 if description is not None and relative_volume != 0 else 0

    def d_relative_volume_up(self, relative_volume_div, description):
        if description is not None and relative_volume_div > 0:
            return relative_volume_div
        else:
            return 0

    def d_relative_volume_down(self, relative_volume_div, description):
        if description is not None and relative_volume_div < 0:
            return abs(relative_volume_div)
        else: 
            return 0

    def relative_volume_div(self, relative_volume, div, description):
        return relative_volume/div -1 if description is not None else 0
        
    def print_data_summary(self):
        for name, day in self.data_dict.items():
            print("==========================================")
            print(name)
            for label, data in day.items():
                print(data.head())
        print("==========================================\n\n\n\n\n")
        print(self.data_dict)


if __name__ == "__main__":
    try:
        model = Model()
    except Exception as e:
        print(e)

            
        