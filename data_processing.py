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

        print("Reading in all data and alerts in:", data_folder)
        with open("raw_data.txt") as files:
            for file in files:
                file = file.replace("\n", "")
                df_temp = pd.read_csv(data_folder + file, header=0) 
                if "A" in file:  # Alert file
                    name = file.replace("A.csv", "")
                    self.data_dict["alerts"][name] = df_temp
                    # print(name)
                else:  # Data file
                    name = file.replace("D.csv", "")
                    self.data_dict["data"][name] = df_temp
                    # print(name)

        self.data = self.data_dict["data"]
        self.calculate_relative_volume_div_by()
        self.calculate_d_relative_volume_up_or_down()
        self.calculate_ADX_filtered_RSI()
        self.calculate_volatility()
        ###################################### End __init__

    def calculate_ADX_filtered_RSI(self):
        print("Calculating ADX Filtered RSI columns...")
        cols_out = [
                "ADX Filtered RSI (7)",
                "ADX Filtered RSI (14)"
            ]
        cols_val = [
                "Average Directional Index (14)",
                "Average Directional Index (14)"
            ]
        cols_left = [
                "Relative Strength Index (7)",
                "Relative Strength Index (14)",
            ]
        cols_right = [
                "Stochastic RSI Fast (3, 3, 14, 14)",
                "Stochastic RSI Slow (3, 3, 14, 14)",
            ]
        cols_to_print = ["Ticker"]
        cols_to_print = cols_to_print + cols_out + cols_val + cols_left + cols_right
        for day, df in self.data.items():
            for col_out, val, left, right in zip(cols_out, cols_val, cols_left, cols_right):
                df[col_out] = df.apply(
                            lambda row: self.ADX_filtered_RSI_lambda(row[val], row[left], row[right], row["Description"]),
                            axis=1)
            print(df[cols_to_print])
        print("Done!")
        
    
    def ADX_filtered_RSI_lambda(self, val, left, right, description):
        if description is not None:
            if val > 20:
                return left
            else:
                return right
        else:
            return 0
    
    def calculate_volatility(self):
        print("Calculating volatility columns...")
        cols_out = [
                "Volatility D/W",
                "Volatility D/M",
                "Volatility W/M",
                "Bollinger Price/Upper-1",
                "Bollinger Price/Lower-1",	
                "Bollinger Upper/Lower Band (20)",
                "Keltner CH Price/Upper-1",
                "Keltner CH Price/Lower-1",	
                "Keltner Channels Upper/Lower Band (20)",
                "Donchian CH Price/Lower-1",
                "Donchian CH Price/Upper-1"
                "Relative Strength Index (7/14)",
                "ADX Filtered RSI (7/14)"
            ]
        print(cols_out)
        numerator_cols = [
                "Volatility",
                "Volatility Week",
                "Volatility Month",
                "Price",
                "Price",
                "Bollinger Upper Band (20)",
                "Price",
                "Price",
                "Keltner Channels Upper Band (20)",
                "Price",
                "Price",
                "Relative Strength Index(7)",
                "ADX Filtered RSI (7)",
            ]
        denominator_cols = [
                "Volatility Week",
                "Volatility Month",
                "Volatility Month",
                "Bollinger Upper Band (20)",
                "Bollinger Lower Band (20)",
                "Bollinger Lower Band (20)",
                "Keltner Channels Upper Band (20)",
                "Keltner Channels Lower Band (20)",
                "Keltner Channels Lower Band (20)",
                "Donchian Channels Lower Band (20)",
                "Donchian Channels Upper Band (20)",
                "Relative Strength Index(14)",
                # "ADX Filtered RSI (14)",
            ]
        cols_to_print = ["Ticker"]
        cols_to_print = cols_to_print + cols_out + numerator_cols + denominator_cols
        for day, df in self.data.items():
            for col_out, num_col, den_col in zip(cols_out, numerator_cols, denominator_cols):
                df[col_out] = df.apply(
                            lambda row: self.volatility_lambda(row[num_col], row[den_col], row["Description"]),
                            axis=1)
            # print(df[cols_to_print])
        print("Done!")

    def volatility_lambda(self, numerator, denominator, description):
        if description is not None and denominator != 0:
            return numerator/denominator - 1
        
    def calculate_relative_volume_div_by(self):
        print("Calculating relative volume div columns...")
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
        print("Done!")

    def calculate_d_relative_volume_up_or_down(self):
        print("Calculating d-relative volume up and down cols")
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
            # print("=======================")
            # print(day)
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
            
            # print(cols_to_print)
            # print(df[cols_to_print])
        print("Done!")
       
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

            
        