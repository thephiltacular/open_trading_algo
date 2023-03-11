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
# print(self.data[day]["Ticker"].head())


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
        self.calculate_under_over_sold()
        self.calculate_UO_trend_D()
        self.calculate_UO_trend_U()
        self.calculate_UO_overbought()
        self.calculate_price_high_low_div_atr()
        self.calculate_CMF_trend_d()
        self.calculate_abs_trend_d()
        self.calculate_momentum_MACD()
        self.calculate_trend_u()
        self.calculate_d_exp_ma()
        self.calculate_d_n_high_low()
        self.calculate_n_div_price()
        self.calculate_fibonacci_min()
        
        ###################################### End __init__
    

    # col RJ
    def calculate_fibonacci_min(self):
        print("Calculating Fibonacci Minimum, Closest D-Pivot, and Current Pivot Level...")
        inputs = [ # r3, r2, r1, p, s1, s2, s3
                ("D-Pivot Fibonacci R3",
                "D-Pivot Fibonacci R2",
                "D-Pivot Fibonacci R1",
                "D-Pivot Fibonacci P",
                "D-Pivot Fibonacci S1",
                "D-Pivot Fibonacci S2",
                "D-Pivot Fibonacci S3")
            ]
        for day, df in self.data.items():
            for r3, r2, r1, p, s1, s2, s3 in inputs:
                self.data[day]["Fibonacci Minimum","Closest D-Pivot", "Current Pivot Level"] = df.apply(
                            lambda row: self.fib_selection_lambda(row[r3],
                                                                  row[r2],
                                                                  row[r1],
                                                                  row[p],
                                                                  row[s1],
                                                                  row[s2],
                                                                  row[s3],
                                                                  row["Description"]),
                            axis=1)
            print(self.data[day]["Fibonacci Minimum","Closest D-Pivot", "Current Pivot Level"].head())

        print("Done!")
        

    def fib_selection_lambda(self, r3, r2, r1, p, s1, s2, s3, description):
        minimum = 0
        if description is not None:
            minimum = min(r3, r2, r1, p, s1, s2, s3)
            if minimum == r3:
                closest = "D-Pivot Fibonacci R3"
                pivot_level = 3
            elif minimum == r2:
                closest = "D-Pivot Fibonacci R2"
                pivot_level = 2
            
            elif minimum == r1:
                closest = "D-Pivot Fibonacci R1"
                pivot_level = 1
            elif minimum == p:
                closest = "D-Pivot Fibonacci P"
                pivot_level = 0
            elif minimum == s1:
                closest = "D-Pivot Fibonacci S1"
                pivot_level = -1
            elif minimum == s2:
                closest = "D-Pivot Fibonacci S2"
                pivot_level = -2
            else:
                closest = "D-Pivot Fibonacci S3"
                pivot_level = -3
            print(minimum, closest, pivot_level)
            return minimum, closest, pivot_level
        else:
            return 0, "", 0
            

    # cols RM, RN
    def calculate_n_div_price(self):
        print("Calculating ATR/Price, ADR/Price, ADR/ATR (Price)....")
        inputs = [  # col_out, numerator, denominator
                ("ATR/Price", "Average True Range (14)", "Price"),
                ("ADR/Price", "Average Day Range (14)", "Price"),
                ("ADR/ATR (Price)", "ADR/Price", "ATR/Price"),
            ]
        cols_to_print = ["ATR/Price", "ADR/Price", "ADR/ATR (Price)"]
        for day, df in self.data.items():
            for col_out, numerator, denominator in inputs:
                self.data[day][col_out] = df.apply(
                            lambda row: self.n_div_lambda(row[numerator], row[denominator], row["Description"]),
                            axis=1)
            print(self.data[day][cols_to_print])
        print("Done!")

    def n_div_lambda(self, numerator, denominator, description):
        if description is not None and denominator != 0:
            return numerator/denominator
        else:
            return 0

    
    # cols RO through RY, RC through RI
    def calculate_d_n_high_low(self):
        print("Calculating the following columns...")
        inputs = [ # col_out, numerator, denominator
                ("D-52 Week High",  "Price", "52 Week High"),
                ("D-6-Month High",  "Price", "6-Month High"),
                ("D-3-Month High",  "Price", "3-Month High"),
                ("D-1-Month High",  "Price", "1-Month High"),
                ("D-52 Week Low",   "Price", "52 Week Low"),
                ("D-6-Month Low",   "Price", "6-Month Low"),
                ("D-3-Month Low",   "Price", "3-Month Low"),
                ("D-1-Month Low",   "Price", "1-Month Low"),
                ("Price/VWMA (20)", "Price", "Volume Weighted Moving Average (20)"),
                ("Price/VWMA",      "Price", "Volume Weighted Moving Average"),
                ("D-Pivot Fibonacci S3", "Price", "Pivot Fibonacci S3"),
                ("D-Pivot Fibonacci S2", "Price", "Pivot Fibonacci S2"),
                ("D-Pivot Fibonacci S1", "Price", "Pivot Fibonacci S1"),
                ("D-Pivot Fibonacci P" , "Price", "Pivot Fibonacci P" ),
                ("D-Pivot Fibonacci R1", "Price", "Pivot Fibonacci R1"),
                ("D-Pivot Fibonacci R2", "Price", "Pivot Fibonacci R2"),
                ("D-Pivot Fibonacci R3", "Price", "Pivot Fibonacci R3"),
                
            ]
        cols_to_print = []
        for col_out, numerator, denominator in inputs:
            cols_to_print += [col_out]
        print(cols_to_print)
        for day, df in self.data.items():
            for col_out, numerator, denominator in inputs:
                self.data[day][col_out] = df.apply(
                            lambda row: self.d_n_lambda(row[numerator], row[denominator], row["Description"]),
                            axis=1)
            # print(self.data[day][cols_to_print])
        print("Done!")

    def d_n_lambda(self, numerator, denominator, description):
        if description is not None and denominator != 0:
            return (numerator/denominator) - 1.0
        else:
            return 0

    # cols NA through NG
    def calculate_d_exp_ma(self):
        print("Calculating the following columns...")
        inputs = [  # col_out, left, right
                ("D-Exponential Moving Average (100/200)", "D-Exponential Moving Average (100)", "D-Exponential Moving Average (200)"),
                ("D-Exponential Moving Average (50/100)", "D-Exponential Moving Average (50)", "D-Exponential Moving Average (100)"),
                ("D-Exponential Moving Average (20/50)", "D-Exponential Moving Average (20)", "D-Exponential Moving Average (50)"),
                ("D-Exponential Moving Average (20/30)", "D-Exponential Moving Average (20)", "D-Exponential Moving Average (30)"),
                ("D-Exponential Moving Average (10/20)", "D-Exponential Moving Average (10)", "D-Exponential Moving Average (20)"),
                ("D-Exponential Moving Average (5/10)", "D-Exponential Moving Average (5)", "D-Exponential Moving Average (10)"),
                ("EMA Gap Slow-Fast", "Slow EMA Avg", "Fast EMA Avg")
            ]
        cols_to_print = []
        for col_out, left, right in inputs:
            cols_to_print += [col_out]
        print(cols_to_print)
        for day, df in self.data.items():
            for col_out, left, right in inputs:
                self.data[day][col_out] = df.apply(
                            lambda row: self.neg_diff_lambda(row[left], row[right]),
                            axis=1)
            # print(self.data[day][cols_to_print])
        print("Done!")

    def neg_diff_lambda(self, left, right):
        return -1.0*(left - right)

    # cols NH through NR
    def calculate_trend_u(self):
        print ("Calculating Trend U for ....")
        inputs = [  # col_out, col
                ("EMA Avg Trend U",   "EMA Gap Slow-Fast"),
                ("EMA (200) Trend U", "D-Exponential Moving Average (200)"),
                ("EMA (100) Trend U", "D-Exponential Moving Average (100)"),
                ("EMA (50) Trend U",  "D-Exponential Moving Average (50)"),
                ("EMA (30) Trend U",  "D-Exponential Moving Average (30)"),
                ("EMA (20) Trend U",  "D-Exponential Moving Average (20)"),
                ("EMA (10) Trend U",  "D-Exponential Moving Average (10)"),
                ("EMA (5) Trend U",   "D-Exponential Moving Average (5)"),
                ("Parabolic Trend U", "D-Parabolic SAR"),
                ("Hull MA Trend U",   "D-Hull Moving Average (9)"),
                ("Ichimoku Trend U",  "Ichimoku Span A/B-1"),       
            ]
        cols_to_print = []
        for col_out, col in inputs:
            cols_to_print += [col_out]
        print(cols_to_print)
        for day, df in self.data.items():
            for col_out, col in inputs:
                self.data[day][col_out] = df.apply(
                            lambda row: self.trend_lambda(row[col], row["Description"]),
                            axis=1)
            print(self.data[day][cols_to_print])
        print("Done!")

    def trend_lambda(self, val, description):
        if description is not None and val >= 0:
            return val
        else:
            return 0

    # col NS
    def calculate_momentum_MACD(self):
        print("Calculating MACD L>S Mom Up...")
        inputs = [ # col_out, lower, upper
                ("MACD L>S Mom Up", "MACD Level (12, 16)", "MACD Signal (12, 16)")
            ]
        for day, df in self.data.items():
            for col_out, lower, upper in inputs:
                self.data[day][col_out] = df.apply(
                            lambda row: self.MACD_lambda(row[upper], row[lower], row["Description"]),
                            axis=1)
                print(self.data[day][col_out])
        print("Done!")
        
    def MACD_lambda(self, upper, lower, description):
        if description is not None and upper > lower and lower != 0:
            return abs(upper)/abs(lower)
        else:
            return 0

    # cols NV through OF
    def calculate_abs_trend_d(self):
        print ("Calculating ABS Trend for ....")
        inputs = [  # col_out, col
                ("EMA Avg Trend D",   "EMA Gap Slow-Fast"),
                ("EMA (200) Trend D", "D-Exponential Moving Average (200)"),
                ("EMA (100) Trend D", "D-Exponential Moving Average (100)"),
                ("EMA (50) Trend D",  "D-Exponential Moving Average (50)"),
                ("EMA (30) Trend D",  "D-Exponential Moving Average (30)"),
                ("EMA (20) Trend D",  "D-Exponential Moving Average (20)"),
                ("EMA (10) Trend D",  "D-Exponential Moving Average (10)"),
                ("EMA (5) Trend D",   "D-Exponential Moving Average (5)"),
                ("Parabolic Trend D", "D-Parabolic SAR"),
                ("Hull MA Trend D",   "D-Hull Moving Average (9)"),
                ("Ichimoku Trend D",  "Ichimoku Span A/B-1"),       
            ]
        cols_to_print = []
        for col_out, col in inputs:
            cols_to_print += [col_out]
        print(cols_to_print)
        for day, df in self.data.items():
            for col_out, col in inputs:
                self.data[day][col_out] = df.apply(
                            lambda row: self.abs_trend_lambda(row[col], row["Description"]),
                            axis=1)
            # print(self.data[day][cols_to_print])
        print("Done!")

    def abs_trend_lambda(self, val, description):
        if description is not None and val < 0:
            return abs(val)
        else:
            return 0

    # col OG
    def calculate_CMF_trend_d(self):
        print("Calculating CMF Trend D...")
        inputs = [  # col_out, val, lower, upper
                ("CMF Trend D", "Money Flow (14)", -0.2, -0.05),
            ]
        cols_to_print = ["Ticker"]
        for col_out, val, lower, upper in inputs:
            cols_to_print += [col_out]
        for day, df in self.data.items():
            for col_out, val, lower, upper in inputs:
                self.data[day][col_out] = df.apply(
                            lambda row: self.CMF_trend_lambda(row[val], lower, upper, row["Description"]),
                            axis=1)
            # print(self.data[day][cols_to_print])
        print("Done!")

    def CMF_trend_lambda(self, val, lower, upper, description):
        if description is not None and val > lower and val < upper:
            return abs(val)
        else:
            return 0

    # cols RZ through SG
    def calculate_price_high_low_div_atr(self):
        print("Calculating Price D-## Week/ATR...")
        inputs = [  # col_out, price, sub, denominator
                ("D-52 Week High/ATR", "Price", "52 Week High", "Average True Range (14)"),
                ("D-6-Month High/ATR", "Price", "6-Month High", "Average True Range (14)"),
                ("D-3-Month High/ATR", "Price", "3-Month High", "Average True Range (14)"),
                ("D-1-Month High/ATR", "Price", "1-Month High", "Average True Range (14)"),
                ("D-52 Week Low/ATR", "Price", "52 Week Low", "Average True Range (14)"),
                ("D-6-Month Low/ATR", "Price", "6-Month Low", "Average True Range (14)"),
                ("D-3-Month Low/ATR", "Price", "3-Month Low", "Average True Range (14)"),
                ("D-1-Month Low/ATR", "Price", "1-Month Low", "Average True Range (14)"),
            ]
        cols_to_print = ["Ticker"]
        for col_out, price, sub, denominator in inputs:
            cols_to_print += [col_out]
        for day, df in self.data.items():
            for col_out, price, sub, denominator in inputs:
                self.data[day][col_out] = df.apply(
                            lambda row: self.price_high_low_div_art_lambda(row[price], row[sub], row[denominator], row["Description"]),
                            axis=1)
            # print(self.data[day][cols_to_print])
        print("Done!")

    def price_high_low_div_art_lambda(self, price, sub, denominator, description):
        if description is not None and denominator != 0:
            return abs((price-sub)/denominator)
        else:
            return 0

    
    def calculate_UO_overbought(self):
        print("Calculating UO overbought...")
        inputs = [  # col_out, price, lower, upper
                ("UO Trend D", "Ultimate Oscillator (7, 14, 28)", 70),
            ]
        cols_to_print = ["Ticker"]
        for col_out, price, lower in inputs:
            cols_to_print += [col_out]
        for day, df in self.data.items():
            for col_out, price, lower in inputs:
                self.data[day][col_out] = df.apply(
                            lambda row: self.UO_trend_overbought_lambda(row[price], lower, row["Description"]),
                            axis=1)
            # print(self.data[day][cols_to_print])
        print("Done!")


    def UO_trend_overbought_lambda(self, price, lower, description):
        if description is not None and price >= lower:
            return price
        else:
            return 0    

    def calculate_UO_trend_U(self):
        print("Calculating UO Trend U...")
        inputs = [  # col_out, price, lower, upper
                ("UO Trend D", "Ultimate Oscillator (7, 14, 28)", 50, 70),
                ("MFI Trend D", "Money Flow (14)", 20, 50),
                ("MFI Trend U", "Money Flow (14)", 50, 80),
                ("CMF Trend U", "Chaikin Money Flow (20)", 0.05, 0.2),
                
            ]
        cols_to_print = ["Ticker"]
        for col_out, price, lower, upper in inputs:
            cols_to_print += [col_out]
        for day, df in self.data.items():
            for col_out, price, lower, upper in inputs:
                self.data[day][col_out] = df.apply(
                            lambda row: self.UO_trend_u_lambda(row[price], lower, upper, row["Description"]),
                            axis=1)
            # print(self.data[day][cols_to_print])
        print("Done!")


    def UO_trend_u_lambda(self, price, lower, upper, description):
        if description is not None and price > lower and price < upper:
            return price
        else:
            return 0


    def calculate_UO_trend_D(self):
        print("Calculating UO Trend D...")
        inputs = [  # col_out, price, lower, upper
                ("UO Trend D", "Ultimate Oscillator (7, 14, 28)", 30, 50),
            ]
        for day, df in self.data.items():
            for col_out, price, lower, upper in inputs:
                self.data[day][col_out] = df.apply(
                            lambda row: self.UO_trend_d_lambda(row[price], lower, upper, row["Description"]),
                            axis=1)
            # print(self.data[day]["UO Trend D"])
        print("Done!")


    def UO_trend_d_lambda(self, price, lower, upper, description):
        if description is not None and price > lower and price <= upper:
            return price
        else:
            return 0

    def calculate_under_over_sold(self):
        print("Calculating under or oversold columns...")
        inputs = [  # col_out, price, value
                ("UO Oversold", "Ultimate Oscillator (7, 14, 28)", 30),
            ]
        for day, df in self.data.items():
            for col_out, price, val in inputs:
                self.data[day][col_out] = df.apply(
                            lambda row: self.under_over_sold_lambda(row[price], val, row["Description"]),
                            axis=1)
            # print(self.data[day]["UO Oversold"])
        print("Done!")

    def under_over_sold_lambda(self, price, value, description):
        if description is not None and price <= value:
            return price
        else:
            return 0
   
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
                self.data[day][col_out] = df.apply(
                            lambda row: self.ADX_filtered_RSI_lambda(row[val], row[left], row[right], row["Description"]),
                            axis=1)
            # print(self.data[day][cols_to_print])
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
                "ADX Filtered RSI (7/14)",
                "Stochastic %K/%D (14, 3, 3)", # SN
                "Stochastic RSI Fast/Slow (3, 3, 14, 14)", # SM
                
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
                "Relative Strength Index (7)",
                "ADX Filtered RSI (7)",
                "Stochastic %K (14, 3, 3)",
                "Stochastic RSI Fast (3, 3, 14, 14)"
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
                "Relative Strength Index (14)",
                "Stochastic %D (14, 3, 3)",
                "Stochastic RSI Slow (3, 3, 14, 14)"
            ]
        cols_to_print = ["Ticker"]
        cols_to_print = cols_to_print + cols_out + numerator_cols + denominator_cols
        for day, df in self.data.items():
            for col_out, num_col, den_col in zip(cols_out, numerator_cols, denominator_cols):
                self.data[day][col_out] = df.apply(
                            lambda row: self.volatility_lambda(row[num_col], row[den_col], row["Description"]),
                            axis=1)
            # print(self.data[day][cols_to_print])
        print("Done!")

    def volatility_lambda(self, numerator, denominator, description):
        if description is not None and denominator != 0:
            return numerator/denominator - 1.0
        else:
            return 0
        
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
            # self.data[day]["D-Relative Volume"] = df.apply(
            #         lambda row: self.d_relative_volume(row["Description"], row["Relative Volume"]),
            #         axis=1
            #     )
            # Calculate 
            for col_out, col in zip(cols_out, cols):
                col_out_name = "Relative Volume" + col_out
                self.data[day][col_out_name] = df.apply(
                    lambda row: self.relative_volume_div(row["Volume"], row[col], row["Description"]),
                    axis=1)
                # percentile_rank_col = col_out_name + "(pct rank)"
                # print("col:", col, "col_out:", col_out_name, "pct_rank:", percentile_rank_col)
                # self.data[day][percentile_rank_col] = self.data[day][col_out_name].rank(pct=True)
                cols_to_print += [col, 
                                    col_out_name, 
                                #   percentile_rank_col
                                    ]
                # print(self.data[day][percentile_rank_col].head())
            
            # print(cols_to_print)
            # print(self.data[day][cols_to_print])
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
            self.data[day]["D-Relative Volume"] = df.apply(
                    lambda row: self.d_relative_volume(row["Relative Volume"], row["Description"]),
                    axis=1
                )
            # Calculate 
            for col_out, col in zip(cols_out, cols):
                # col_out_name = "Relative Volume" + col_out
                # up = True 
                if "Up" in col_out:
                    # up = False
                    self.data[day][col_out] = df.apply(
                        lambda row: self.d_relative_volume_up(row[col], row["Description"]),
                        axis=1)
                else:
                    self.data[day][col_out] = df.apply(
                        lambda row: self.d_relative_volume_down(row[col], row["Description"]),
                        axis=1)
                cols_to_print += [col, col_out]
                # print(self.data[day][percentile_rank_col].head())
            
            # print(cols_to_print)
            # print(self.data[day][cols_to_print])
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

            
        