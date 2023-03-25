#! /usr/bin/env python3

import pandas as pd
import numpy as np

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

class Model():
    def __init__(self):
        data_folder = "data/"
        self.data_dict = {
            "data": {},"alerts": {}
        }

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
        self.step = 0
        self.total_steps = 29
        self.calculate_all_model_v8()
        # self.calculate_avg_os()
        # self.calculate_OB()

    # print separator
    def ps(self):
        print("===============================")
        print("Step: ", self.step, "/", self.total_steps)
        self.step += 1

    def calculate_all_model_v8(self):
        self.total_steps = 29
        # calculations from v8 of model
        # 0:
        self.calculate_volume_div()  # cols vm through vv
        self.calculate_d_relative_volume_up_or_down()  # cols UJ through UT and VA through VL
        self.calculate_avg_volume()  # cols UD through UI and UU through UZ
        self.calculate_percent_rank_avg_volume()  # cols UW through UX, UF through UG
        self.calculate_avg_pr()  # cols UD through UE, UU through UV
        # 5:
        self.calculate_ADX_filtered_RSI()
        self.calculate_volatility()  # cols TK through TS
        self.calculate_price_high_low_div_atr() # cols TB through TI
        self.calculate_d_n_high_low()  # cols SJ through SP, ST through TA
        self.calculate_fibonacci_min()  # cols SQ through SS
        # 10:
        self.calculate_UO_OS()  #col SD through SG
        self.calculate_UO_TD()  # col MM
        self.calculate_UO_trend_U() # cols MJ through MK
        self.caclulate_williams()  # col SC
        self.calculate_trend_OS()  # cols RW through RZ
        # 15:
        self.calculate_PR()  # cols RJ through RT
        self.calculate_OB()  # cols SH through SI
        self.calculate_PR_for_RU_RV()  # cols RU through RV
        self.calculate_avg_os()  # cols RH through RI
        self.calculate_PR_os()  # cols RF through RG
        # 20:
        self.calculate_avg_PR_OS_final()  # cols RD through RE
        self.calculate_UO_overbought()  # cols QU through RA
        self.calculate_trend_OB() # cols QQ through QT
        self.calculate_PR_ob()  # cols QD through QP
        self.calculate_avg_ob() # cols QB through QC
        # 25:
        self.calculate_PR_ob()  # cols PZ through QA
        self.calculate_avg_PR_OB_final()  # cols PX through PY
        self.calculate_trend_MD()  # cols PB through PE
        self.calculate_trend_MD_2()  # colds OX through PA
        self.calculate_avg_md()  # cols OP through OW
        
        
    def calculate_all_model_v7(self): 
        # calculation order from v7 of model
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
        self.calculate_EMA_avgs()
        self.calculate_d_n_high_low()
        self.calculate_d_exp_ma()
        self.calculate_abs_trend_d()
        self.calculate_momentum_MACD()
        self.calculate_trend_u()
        self.calculate_n_div_price()
        self.calculate_fibonacci_min()

        # Exporting all calculated data in correct order to CSV by day
        # self.export_calculated_data()
        
        ###################################### End __init__
        

    def calculate_EMA_avgs(self):
        print("Calculating EMA avgs...")
        inputs = [
                ("Fast EMA Avg", "D-Exponential Moving Average (5)", "D-Exponential Moving Average (10)", "D-Exponential Moving Average (20)"),
                ("Slow EMA Avg", "D-Exponential Moving Average (30)", "D-Exponential Moving Average (50)", "D-Exponential Moving Average (100)")
            ]
        cols_to_print = ["Ticker"]
        for col_out, one, two, three in inputs:
            cols_to_print += [col_out]
        print(cols_to_print)
        for day, df in self.data.items():
            for col_out, one, two, three in inputs:
                self.data[day][col_out] = df.apply(
                            lambda row: self.ema_avg_lambda(row[one], row[two], row[three], row["Description"]),
                            axis=1)
            # print(self.data[day][cols_to_print].head())

        print("Done!")

    def ema_avg_lambda(self, one, two, three, description):
        if description is not None and one != 0 and two != 0 and three != 0:
            return (one + two + three)/3.0
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
            # print(self.data[day][cols_to_print])
        print("Done!")

    def trend_lambda(self, val, description):
        if description is not None and val >= 0:
            return val
        else:
            return 0

    ######################################### v8
    # cols 
    def calculate_momentum_MACD(self):
        print("Calculating MACD L>S Mom Up...")
        inputs = [ # col_out, lower, upper
                ("MACD L>S Mom Up", "MACD Level (12, 26)", "MACD Signal (12, 26)")
            ]
        for day, df in self.data.items():
            for col_out, lower, upper in inputs:
                self.data[day][col_out] = df.apply(
                            lambda row: self.MACD_lambda(row[upper], row[lower], row["Description"]),
                            axis=1)
                # print(self.data[day][col_out])
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
    # 29
    ######################################### v8
    # cols OP through OW
    def calculate_avg_md(self):
        self.ps()
        print("Calculating Avg MD....")
        inputs = [ # col_out, cols 1:10, # OB
            "Avg MD",
            "# MD",  # col_out with count of above 0
            "PR-Aroon MD",
            "PR-ADX MD",
            "PR-D-DMI MD ",
            "PR-AO MD",
            "PR-RSI (14) 30-50 MD",
            "PR-RSI (7) 30-50 MD",
            "PR-St-RSI Fast MD",
            "PR-St-RSI Slow MD",
        ]

        print("Using cols:")
        for i in range(1,len(inputs)):
            # print("inputs[", i, "]")
            print(i, "=", inputs[i])

        for day, df in self.data.items():
            #               col_out     count
            self.data[day][[inputs[0], inputs[1]]] = df.apply(
                lambda row: self.avg_lambda(
                    row[inputs[2]],
                    row[inputs[3]],
                    row[inputs[4]],
                    row[inputs[5]],
                    row[inputs[6]],
                    row[inputs[7]],
                    row[inputs[8]],
                    row[inputs[9]],
                    description=row["Description"]
                ), axis=1, result_type="expand")
            # print(self.data[day][[inputs[0], inputs[11]]])
        print("Done!")



    # 28
    ######################################### v8
    # cols OX through PA
    def calculate_trend_MD_2(self):
        self.ps()
        print("Calculating MD columns...")
        inputs = [
            ("AO MD",     "Awesome Oscillator",  None, None),
            ("D-DMI MD ", "Positive Directional Indicator (14)", "Negative Directional Indicator (14)", None),
            ("ADX MD",    "Positive Directional Indicator (14)", "Negative Directional Indicator (14)", "Average Directional Index (14)"),
            ("Aroon MD", "Aroon Down (14)", "Aroon Up (14)", None),
        ]
        
        cols_to_print = ["Ticker"]
        for col_out, val, idx, lower, upper in inputs:
            cols_to_print += [col_out]
        print(cols_to_print)
        for day, df in self.data.items():
            for col_out, left, right, ret in inputs:
                self.data[day][col_out] = df.apply(
                    lambda row: self.trend_md_rsi_lambda(
                        row[left], 
                        row[right] if right is not None else 0, 
                        row[ret] if ret is not None else ret, 
                        row["Description"]),
                    axis=1
                )

    def trend_md_lamda(self, left, right, ret, description):
        if right is None:
            right = 0
        if description is not None:
            if (left - right) < 0:
                if ret is None:
                    return abs(left-right)
                else:
                    return abs(ret)
            else:
                return 0
                
        
    # 27
    ######################################### v8
    # cols PB through PE
    def calculate_trend_MD(self):
        self.ps()
        print("Calculating MD columns...")
        inputs_1 = [  # col_out, val, idx, upper, lower,  
                ("RSI (7) 30-50 MD", "Relative Strength Index (7)",  "Average Directional Index (14)", 30.0, 20.0),
                ("RSI (14) 30-50 MD", "Relative Strength Index (14)", "Average Directional Index (14)", 30.0, 20.0),
            ]
        inputs_2 = [  # col_out, val, idx, upper, lower,  
                ("St-RSI Fast MD", "Stochastic RSI Fast (3, 3, 14, 14)", "Average Directional Index (14)", 50.0, 20.0),
                ("St-RSI Slow MD", "Stochastic RSI Slow (3, 3, 14, 14)", "Average Directional Index (14)", 50.0, 20.0),
            ]
        cols_to_print = ["Ticker"]
        for col_out, val, idx, lower, upper in inputs_1:
            cols_to_print += [col_out]
        for col_out, val, idx, lower, upper in inputs_2:
            cols_to_print += [col_out]
        print(cols_to_print)
        # Calculate RSI
        for day, df in self.data.items():
            for col_out, val, idx, upper, lower in inputs_1:
                self.data[day][col_out] = df.apply(
                            lambda row: self.trend_md_rsi_lambda(row[val], row[idx], upper, lower, row["Description"]),
                            axis=1)
            # print(self.data[day]["UO Oversold"])
        # Calculate ST-RSI
        for day, df in self.data.items():
            for col_out, val, idx, upper, lower in inputs_2:
                self.data[day][col_out] = df.apply(
                            lambda row: self.trend_md_st_rsi_lambda(row[val], row[idx], upper, lower, row["Description"]),
                            axis=1)
            # print(self.data[day]["UO Oversold"])
        print("Done!")

    def trend_md_st_rsi_lambda(self, val, idx, upper, lower, description):
        if description is not None and val > lower and idx <= 20.0 and val < upper:
            return abs(val - upper)
        else:
            return 0

    def trend_md_rsi_lambda(self, val, idx, upper, lower, description):
        if description is not None and val > upper and idx > lower and val < 50.0:
            return abs(val - 50.0)
        else:
            return 0



    # 26
    ######################################### v8
    # cols PX through PY
    def calculate_avg_PR_OB_final(self):
        self.ps()
        print("Calculating avg PR of OB and final PR-OB....")
        inputs = [ # col_out, cols 1:2, col_out_pr,
            "Avg-#OB"
            "PR-Avg OB",
            "PR-# OB",  # col_out with count of above 0
            "PR-OB",
        ]

        print("Using cols:")
        for i in range(1,len(inputs)):
            # print("inputs[", i, "]")
            print(i, "=", inputs[i])

        for day, df in self.data.items():
            #               col_out     count
            self.data[day][inputs[0]] = df.apply(
                lambda row: self.avg_PR_OS_lambda(
                    row[inputs[1]],
                    row[inputs[2]],
                    description=row["Description"]
                ), axis=1)  #, result_type="expand")
            # Calculate PR of the column that we just calculated
            temp = self.data[day][inputs[0]].replace(0, np.nan)
            self.data[day][inputs[3]] = temp.rank(pct=True).replace(np.nan, 0)
            
        print("Done!")


    # 25
    ######################################### v8
    # cols PZ through QA
    def calculate_PR_ob_avg(self):
        self.ps()
        print("Calculating the following Percent Rank Columns...")
        inputs = [  # col_out, col
           ("PR-# OB", "# OB"),
           ("PR-Avg OB", "Avg OB"),
        ]
        cols_to_print = ["Ticker"]
        for col_out, col in inputs:
            cols_to_print += [col_out]
        print(cols_to_print)
        for day, df in self.data.items():
            for col_out, col in inputs:
                temp = df[col].replace(0, np.nan)
                self.data[day][col_out] = temp.rank(pct=True).replace(np.nan, 0)
            # print(self.data[day]["UO Oversold"])
        print("Done!")

    # 24
    ######################################### v8
    # cols QB through QC
    def calculate_avg_ob(self):
        self.ps()
        print("Calculating Avg OB....")
        inputs = [ # col_out, cols 1:10, # OB
            "Avg OB",
            "# OB",  # col_out with count of above 0
            "PR-RSI (7) OB",    
            "PR-RSI (14) OB",   
            "PR-St-RSI Fast OB",
            "PR-St-RSI Slow OB",
            "PR-Stoch K% OB",   
            "PR-Stoch D% OB",   
            "PR-Williams% OB",  
            "PR-CCI OB",        
            "PR-CMF OB",        
            "PR-MFI OB",        
            "PR-UO OB",
            "PR-Bollinger Price/Lower-1 OB",
            "PR-Keltner Price/Lower-1 OB"
        ]

        print("Using cols:")
        for i in range(1,len(inputs)):
            # print("inputs[", i, "]")
            print(i, "=", inputs[i])

        for day, df in self.data.items():
            #               col_out     count
            self.data[day][[inputs[0], inputs[1]]] = df.apply(
                lambda row: self.avg_lambda(
                    row[inputs[2]],
                    row[inputs[3]],
                    row[inputs[4]],
                    row[inputs[5]],
                    row[inputs[6]],
                    row[inputs[7]],
                    row[inputs[8]],
                    row[inputs[9]],
                    row[inputs[10]],
                    row[inputs[11]],
                    row[inputs[12]],
                    row[inputs[13]],
                    row[inputs[14]],
                    description=row["Description"]
                ), axis=1, result_type="expand")
            # print(self.data[day][[inputs[0], inputs[11]]])
        print("Done!")


    # 23
    ######################################### v8
    # cols QD through QP
    def calculate_PR_ob(self):
        self.ps()
        print("Calculating the following Percent Rank Columns...")
        inputs = [  # col_out, col
            ("PR-RSI (7) OB",                 "RSI (7) OB",),
            ("PR-RSI (14) OB",                "RSI (14) OB",),
            ("PR-St-RSI Fast OB",             "St-RSI Fast OB",),
            ("PR-St-RSI Slow OB",             "St-RSI Slow OB",),
            ("PR-Stoch K% OB",                "Stoch K% OB",),
            ("PR-Stoch D% OB",                "Stoch D% OB",),
            ("PR-Williams% OB",               "Williams% OB",),
            ("PR-CCI OB",                     "CCI OB",),
            ("PR-CMF OB",                     "CMF OB",),
            ("PR-MFI OB",                     "MFI OB",),
            ("PR-UO OB",                      "UO OB",),
            ("PR-Bollinger Price/Upper-1 OB", "Bollinger Price/Upper-1 OB",),
            ("PR-Keltner Price/Upper-1 OB",   "Keltner Price/Upper-1 OB",),
        ]
        cols_to_print = ["Ticker"]
        for col_out, col in inputs:
            cols_to_print += [col_out]
        print(cols_to_print)
        for day, df in self.data.items():
            for col_out, col in inputs:
                temp = df[col].replace(0, np.nan)
                self.data[day][col_out] = temp.rank(pct=True)
            # print(self.data[day]["UO Oversold"])
        print("Done!")

    # 22
    ######################################### v8
    # cols QQ through QT
    def calculate_trend_OB(self):
        self.ps()
        print("Calculating OB columns...")
        inputs_1 = [  # col_out, val, idx, upper, lower,  
                ("RSI (7) OB",     "Relative Strength Index (7)",        "Average Directional Index (14)", 70.0, 20.0),
                ("RSI (14) OB",    "Relative Strength Index (14)",       "Average Directional Index (14)", 70.0, 20.0),
            ]
        inputs_2 = [  # col_out, val, idx, upper, lower,  
                ("St-RSI Fast OB", "Stochastic RSI Fast (3, 3, 14, 14)", "Average Directional Index (14)", 80.0, 20.0),
                ("St-RSI Slow OB", "Stochastic RSI Slow (3, 3, 14, 14)", "Average Directional Index (14)", 80.0, 20.0),
            ]
        cols_to_print = ["Ticker"]
        for col_out, val, idx, lower, upper in inputs_1:
            cols_to_print += [col_out]
        for col_out, val, idx, lower, upper in inputs_2:
            cols_to_print += [col_out]
        print(cols_to_print)
        # Calculate RSI
        for day, df in self.data.items():
            for col_out, val, idx, upper, lower in inputs_1:
                self.data[day][col_out] = df.apply(
                            lambda row: self.trend_os_rsi_lambda(row[val], row[idx], upper, lower, row["Description"]),
                            axis=1)
            # print(self.data[day]["UO Oversold"])
        # Calculate ST-RSI
        for day, df in self.data.items():
            for col_out, val, idx, upper, lower in inputs_2:
                self.data[day][col_out] = df.apply(
                            lambda row: self.trend_ob_st_rsi_lambda(row[val], row[idx], upper, lower, row["Description"]),
                            axis=1)
            # print(self.data[day]["UO Oversold"])
        print("Done!")

    def trend_ob_st_rsi_lambda(self, val, idx, upper, description):
        if description is not None and val > upper and idx <= val:
            return val - upper
        else:
            return 0

    def trend_ob_rsi_lambda(self, val, idx, upper, lower, description):
        if description is not None and val > upper and idx > lower:
            return val - upper
        else:
            return 0

    # 21
    ######################################### v8
    # cols QU through RA
    def calculate_UO_overbought(self):
        self.ps()
        print("Calculating UO overbought...")
        inputs = [  # col_out, price, lower
                ("UO OB",       "Ultimate Oscillator (7, 14, 28)", 70.0),
                ("MFI OB",      "Money Flow (14)", 80.0),
                ("CMF OB",      "Chaikin Money Flow (20)", 0.2),
                ("CCI OB",      "Commodity Channel Index (20)", 100.0),
                ("Willams% OB", "Williams Percent Range (14)", -20.0),
                ("Stoch D% OB", "Stochastic %D (14, 3, 3)", 80.0),
                ("Stoch K% OB", "Stochastic %K (14, 3, 3)", 80.0),
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
            return price - lower
        else:
            return 0    


    # 20
    ######################################### v8
    # cols RD through RE
    def calculate_avg_PR_OS_final(self):
        self.ps()
        print("Calculating avg PR of OS and final PR-OS....")
        inputs = [ # col_out, cols 1:2, col_out_pr,
            "Avg-#OS"
            "PR-Avg OS",
            "PR-# OS", 
            "PR-OS",
        ]

        print("Using cols:")
        for i in range(0,len(inputs)):
            # print("inputs[", i, "]")
            print(i, "=", inputs[i])

        for day, df in self.data.items():
            #               col_out     count
            self.data[day][inputs[0]] = df.apply(
                lambda row: self.avg_PR_OS_lambda(
                    row[inputs[1]],
                    row[inputs[2]],
                    description=row["Description"]
                ), axis=1)  #, result_type="expand")
            # Calculate PR of the column that we just calculated
            temp = self.data[day][inputs[0]].replace(0, np.nan)
            self.data[day][inputs[3]] = temp.rank(pct=True).replace(np.nan, 0)
            
        print("Done!")

    def avg_PR_OS_lambda(self, *args, description):
        if description is not None:
            count = 0.0
            sum = 0.0
            for arg in args:
                if arg > 0:
                    sum += arg
                    count += 1.0
            if count != 0:
                return (sum/count)
            else:
                return 0
        else:
            return 0


    # 19
    ######################################### v8
    # cols RF through RG
    def calculate_PR_os(self):
        self.ps()
        print("Calculating the following Percent Rank Columns...")
        inputs = [  # col_out, col
           ("PR-# OS", "# OS"),
           ("PR-Avg OS", "Avg OS"),
        ]
        cols_to_print = ["Ticker"]
        for col_out, col in inputs:
            cols_to_print += [col_out]
        print(cols_to_print)
        for day, df in self.data.items():
            for col_out, col in inputs:
                temp = df[col].replace(0, np.nan)
                self.data[day][col_out] = temp.rank(pct=True).replace(np.nan, 0)
            # print(self.data[day]["UO Oversold"])
        print("Done!")

    # 18
    ######################################### v8
    # cols RH through RI
    def calculate_avg_os(self):
        self.ps()
        print("Calculating Avg OS....")
        inputs = [ # col_out, cols 1:10, # OS
            "Avg OS",
            "# OS",  # col_out with count of above 0
            "PR-RSI (7) OS",    
            "PR-RSI (14) OS",   
            "PR-St-RSI Fast OS",
            "PR-St-RSI Slow OS",
            "PR-Stoch K% OS",   
            "PR-Stoch D% OS",   
            "PR-Williams% OS",  
            "PR-CCI OS",        
            "PR-CMF OS",        
            "PR-MFI OS",        
            "PR-UO OS",
            "PR-Bollinger Price/Lower-1 OS",
            "PR-Keltner Price/Lower-1 OS"
        ]

        print("Using cols:")
        for i in range(1,len(inputs)):
            # print("inputs[", i, "]")
            print(i, "=", inputs[i])

        for day, df in self.data.items():
            #               col_out     count
            self.data[day][[inputs[0], inputs[1]]] = df.apply(
                lambda row: self.avg_lambda(
                    row[inputs[2]],
                    row[inputs[3]],
                    row[inputs[4]],
                    row[inputs[5]],
                    row[inputs[6]],
                    row[inputs[7]],
                    row[inputs[8]],
                    row[inputs[9]],
                    row[inputs[10]],
                    row[inputs[11]],
                    row[inputs[12]],
                    row[inputs[13]],
                    row[inputs[14]],
                    description=row["Description"]
                ), axis=1, result_type="expand")
            # print(self.data[day][[inputs[0], inputs[11]]])
        print("Done!")
        
    def avg_lambda(self, *args, description):
        if description is not None:
            count = 0.0
            sum = 0.0
            for arg in args:
                if arg > 0:
                    sum += arg
                    count += 1.0
            if count != 0:
                return (sum/count), count
            else:
                return 0, 0
        else:
            return 0, 0

    # 17
    ######################################### v8
    # cols RU through RV
    def calculate_PR_for_RU_RV(self):
        self.ps()
        print("Calculating the following Percent Rank Columns...")
        inputs = [  # col_out, col
            ("PR-Bollinger Price/Lower-1 OS", "Bollinger Price/Lower-1 OS" ),
            ("PR-Keltner Price/Lower-1 OS",   "Keltner Price/Lower-1 OS" ),
        ]
        cols_to_print = ["Ticker"]
        for col_out, col in inputs:
            cols_to_print += [col_out]
        print(cols_to_print)
        for day, df in self.data.items():
            for col_out, col in inputs:
                temp = df[col].replace(0, np.nan)
                self.data[day][col_out] = temp.rank(pct=True)
                # self.data[day][col_out] = df.apply(
                #             lambda row: self.trend_os_st_rsi_lambda(row[val], row[idx], upper, row["Description"]),
                #             axis=1)
            # print(self.data[day]["UO Oversold"])
        print("Done!")

    # 16
    ######################################### v8
    # cols SH through SI, RB through RC
    def calculate_OB(self):
        self.ps()
        print("Calculating the following columns...")
        inputs = [  #col_out, col
            ("Bollinger Price/Lower-1 OS", "Bollinger Price/Lower-1"),
            ("Keltner Price/Lower-1 OS", "Keltner Price/Lower-1"),
            ("Bollinger Price/Upper-1 OS", "Bollinger Price/Upper-1"),
            ("Keltner Price/Upper-1 OS", "Keltner Price/Upper-1"),
            
        ]
        cols_to_print = ["Ticker"]
        for col_out, col in inputs:
            cols_to_print += [col_out]
        print(cols_to_print)
        for day, df in self.data.items():
            for col_out, col in inputs:
                temp = df[col].replace(0, np.nan)
                quartile = temp.quantile(0.25)
                self.data[day][col_out] = df.apply(
                            lambda row: self.quartile_lambda(row[col], quartile, row["Description"]),
                            axis=1)
                
                
    def quartile_lambda(self, col, quartile, description):
        if description is not None and col != 0 and col <= quartile:
            return col
        else:
            return 0
                
    
    # 15
    ######################################### v8
    # cols RJ through RT
    def calculate_PR(self):
        self.ps()
        print("Calculating the following Percent Rank Columns...")
        inputs = [  # col_out, col
            ("PR-RSI (7) OS",                  "RSI (7) OS",),
            ("PR-RSI (14) OS",                 "RSI (14) OS",),
            ("PR-St-RSI Fast OS",              "St-RSI Fast OS",),
            ("PR-St-RSI Slow OS",              "St-RSI Slow OS",),
            ("PR-Stoch K% OS",                 "Stoch K% OS",),
            ("PR-Stoch D% OS",                 "Stoch D% OS",),
            ("PR-Williams% OS",                "Williams% OS",),
            ("PR-CCI OS",                      "CCI OS",),
            ("PR-CMF OS",                      "CMF OS",),
            ("PR-MFI OS",                      "MFI OS",),
            ("PR-UO OS",                       "UO OS",),
        ]
        cols_to_print = ["Ticker"]
        for col_out, col in inputs:
            cols_to_print += [col_out]
        print(cols_to_print)
        for day, df in self.data.items():
            for col_out, col in inputs:
                temp = df[col].replace(0, np.nan)
                self.data[day][col_out] = temp.rank(pct=True)
                # self.data[day][col_out] = df.apply(
                #             lambda row: self.trend_os_st_rsi_lambda(row[val], row[idx], upper, row["Description"]),
                #             axis=1)
            # print(self.data[day]["UO Oversold"])
        print("Done!")
        

    def percent_rank_lamda(self, val, description):
        if description is not None:
            return val
        else:
            return 0

    # 14
    ######################################### v8
    # cols RW through RZ
    def calculate_trend_OS(self):
        self.ps()
        print("Calculating under or oversold columns...")
        inputs_1 = [  # col_out, val, idx, lower, upper 
                ("RSI (7) OS",      "Relative Strength Index (7)", "Average Directional Index (14)", 20.0, 30.0),
                ("RSI (14) OS",     "Relative Strength Index (14)", "Average Directional Index (14)", 20.0, 30.0),
            ]
        inputs_2 = [  # col_out, val, idx, upper
                ("St-RSI Fast OS",  "Stochastic RSI Fast (3, 3, 14, 14)", "Average Directional Index (14)", 20.0),
                ("St-RSI Slow OS",  "Stochastic RSI Slow (3, 3, 14, 14)", "Average Directional Index (14)", 20.0),
            ]
        cols_to_print = ["Ticker"]
        for col_out, val, idx, lower, upper in inputs_1:
            cols_to_print += [col_out]
        # print(cols_to_print)
        for col_out, val, idx, upper in inputs_2:
            cols_to_print += [col_out]
        print(cols_to_print)
        # Calculate RSI
        for day, df in self.data.items():
            for col_out, val, idx, lower, upper in inputs_1:
                self.data[day][col_out] = df.apply(
                            lambda row: self.trend_os_rsi_lambda(row[val], row[idx], upper, lower, row["Description"]),
                            axis=1)
            # print(self.data[day]["UO Oversold"])
        # Calculate ST-RSI
        for day, df in self.data.items():
            for col_out, val, idx, upper in inputs_2:
                self.data[day][col_out] = df.apply(
                            lambda row: self.trend_os_st_rsi_lambda(row[val], row[idx], upper, row["Description"]),
                            axis=1)
            # print(self.data[day]["UO Oversold"])
        print("Done!")

    def trend_os_st_rsi_lambda(self, val, idx, upper, description):
        if description is not None and val < upper and idx <= val:
            return abs(val-upper)
        else:
            return 0

    def trend_os_rsi_lambda(self, val, idx, upper, lower, description):
        if description is not None and val < upper and idx > lower:
            return abs(val - upper)
        else:
            return 0

    # 13
    ######################################### v8
    # cols SC
    def caclulate_williams(self):
        self.ps()
        print("Calculating Williams% OS....")
        inputs = [  # col_out, price, lower, upper
                ("Williams% OS", "Williams Percent Range (14)", -100.0, -80.0),
            ]
        for day, df in self.data.items():
            for col_out, price, lower, upper in inputs:
                self.data[day][col_out] = df.apply(
                            lambda row: self.williams_lamda(row[price], lower, upper, row["Description"]),
                            axis=1)
            # print(self.data[day]["UO Trend D"])
        print("Done!")

    def williams_lamda(self, val, lower, upper, description):
        if description is not None and val >= lower and val <= upper:
            return abs(val - upper)
        else:
            return 0


    # 12
    ######################################### v8
    # cols MJ through MK
    def calculate_UO_trend_U(self):
        self.ps()
        print("Calculating UO Trend U...")
        inputs = [  # col_out, price, lower, upper
                # ("UO Trend D", "Ultimate Oscillator (7, 14, 28)", 50, 70),
                ("MFI TD", "Money Flow (14)", 20, 50),
                ("MFI TU", "Money Flow (14)", 50, 80),
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

    # 11
    ######################################### v8
    # cols MM
    def calculate_UO_TD(self):
        self.ps()
        print("Calculating UO Trend D...")
        inputs = [  # col_out, price, lower, upper
                ("UO TD", "Ultimate Oscillator (7, 14, 28)", 30.0, 50.0),
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


            # ("PR-OS",                          ),
            # ("Avg-# OS",                       ),
            # ("PR-# OS",                        ),
            # ("PR-Avg OS",                      ),
            # ("# OS",                           ),
            # ("Avg OS",                         ),


    # 10
    ######################################### v8
    # cols SD through SG
    def calculate_UO_OS(self):
        self.ps()
        print("Calculating under or oversold columns...")
        inputs = [  # col_out, price, value
                ("UO OS", "Ultimate Oscillator (7, 14, 28)", 30.0),
                ("MFI OS", "Money Flow (14)", 20.0),
                ("CMF OS", "Chaikin Money Flow (20)", -0.2),
                ("CCI OS", "Commodity Channel Index (20)", -100.0),
                # ("Williams% OS", "Commodity Channel Index (20)", -100.0),
                # ("RSI (7) OS", "Commodity Channel Index (20)", -100.0),
                # ("RSI (14) OS", "Commodity Channel Index (20)", -100.0),
                # ("St-RSI Fast OS", "Commodity Channel Index (20)", -100.0),
                # ("St-RSI Slow OS", "Commodity Channel Index (20)", -100.0),
                ("Stoch K% OS", "Stochastic %K (14, 3, 3)", 20.0),
                ("Stoch D% OS", "Stochastic %D (14, 3, 3)", 20.0),
                
            ]
        cols_to_print = ["Ticker"]
        for col_out, price, val in inputs:
            cols_to_print += [col_out]
        print(cols_to_print)
        for day, df in self.data.items():
            for col_out, price, val in inputs:
                self.data[day][col_out] = df.apply(
                            lambda row: self.under_over_sold_lambda(row[price], val, row["Description"]),
                            axis=1)
            # print(self.data[day]["UO Oversold"])
        print("Done!")

    def under_over_sold_lambda(self, price, value, description):
        if description is not None and price <= value and value !=0:
            return abs(price - value)
        else:
            return 0


    # 9
    ######################################### v8
    # col SQ through SS
    def calculate_fibonacci_min(self):
        self.ps()
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
                self.data[day][["Fibonacci Minimum", "Closest D-Pivot", "Current Pivot Level"]] = df.apply(
                            lambda row: self.fib_selection_lambda(row[r3],
                                                                  row[r2],
                                                                  row[r1],
                                                                  row[p],
                                                                  row[s1],
                                                                  row[s2],
                                                                  row[s3],
                                                                  row["Description"]),
                            axis=1, result_type="expand")
            # print(self.data[day][["Fibonacci Minimum","Closest D-Pivot", "Current Pivot Level"]].head())

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
            # print(minimum, closest, pivot_level)
            # row["Fibonacci Minimum"] = minimum
            # row["Closest D-Pivot"] = closest
            # row["Current Pivot Level"] = pivot_level
            return minimum, closest, pivot_level
        else:
            # row["Fibonacci Minimum"] = 0
            # row["Closest D-Pivot"] = ""
            # row["Current Pivot Level"] = 0
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
            # print(self.data[day][cols_to_print])
        print("Done!")

    def n_div_lambda(self, numerator, denominator, description):
        if description is not None and denominator != 0:
            return numerator/denominator
        else:
            return 0

    # 8
    ######################################### v8
    # cols SJ through SP, ST through TA
    def calculate_d_n_high_low(self):
        self.ps()
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
                ("Price/VWMA",      "Price", "Volume Weighted Average Price"),
                ("D-Pivot Fibonacci S3", "Price", "Pivot Fibonacci S3"),
                ("D-Pivot Fibonacci S2", "Price", "Pivot Fibonacci S2"),
                ("D-Pivot Fibonacci S1", "Price", "Pivot Fibonacci S1"),
                ("D-Pivot Fibonacci P" , "Price", "Pivot Fibonacci P" ),
                ("D-Pivot Fibonacci R1", "Price", "Pivot Fibonacci R1"),
                ("D-Pivot Fibonacci R2", "Price", "Pivot Fibonacci R2"),
                ("D-Pivot Fibonacci R3", "Price", "Pivot Fibonacci R3"),
                ("D-Hull Moving Average (9)", "Price", "Hull Moving Average (9)"),
                ("D-Ichimoku Leading Span A (9, 26, 52, 26)", "Price", "Ichimoku Leading Span A (9, 26, 52, 26)"),
                ("D-Ichimoku Leading Span B (9, 26, 52, 26)", "Price", "Ichimoku Leading Span B (9, 26, 52, 26)"),
                ("Ichimoku Span A/B-1", "Ichimoku Leading Span A (9, 26, 52, 26)", "Ichimoku Leading Span B (9, 26, 52, 26)"),
                ("D-Parabolic SAR", "Price", "Parabolic SAR")
                
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

    # 7
    ######################################### v8
    # cols TB through TI
    def calculate_price_high_low_div_atr(self):
        self.ps()
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

    

   
    # 6
    ######################################### v8
    # cols TK through TS, MN through MV
    def calculate_volatility(self):
        self.ps()
        print("Calculating the following volatility columns...")
        inputs = [
            ("Volatility D/W", "Volatility", "Volatility Week"),
            ("Volatility D/W", "Volatility", "Volatility Week"),
            ("Volatility D/M", "Volatility", "Volatility Month"),
            ("Volatility W/M", "Volatility Week", "Volatility Month"),
            ("Keltner Price/Upper-1", "Price", "Keltner Channels Upper Band (20)"),
            ("Keltner Price/Lower-1", "Price", "Keltner Channels Lower Band (20)"),
            ("Keltner Upper/Lower Band", "Keltner Channels Upper Band (20)","Keltner Channels Lower Band (20)"),
            ("Bollinger Upper/Lower Band", "Bollinger Upper Band (20)","Bollinger Lower Band (20)"),
            ("Bollinger Price/Upper-1", "Price", "Bollinger Upper Band (20)"),
            ("Bollinger Price/Lower-1", "Price", "Bollinger Lower Band (20)"),
            ("Donchian Price/Lower-1", "Price", "Donchian Channels Lower Band (20)"),
            ("Donchian Price/Upper-1", "Price", "Donchian Channels Upper Band (20)"),
            ("Relative Strength Index (7/14)", "Relative Strength Index (7)","Relative Strength Index (14)"),
            ("ADX Filtered RSI (7/14)", "ADX Filtered RSI (7)","Stochastic %D (14, 3, 3)"),
            ("Stochastic %K/%D (14, 3, 3)", "Stochastic %K (14, 3, 3)","Stochastic RSI Slow (3, 3, 14, 14)"),
            ("Stochastic RSI Fast/Slow (3, 3, 14, 14)", "Stochastic RSI Fast (3, 3, 14, 14)","Stochastic RSI Slow (3, 3, 14, 14)"),
            ("D-Exponential Moving Average (5)",   "Price", "Exponential Moving Average (5)"),
            ("D-Exponential Moving Average (10)",  "Price", "Exponential Moving Average (10)"),
            ("D-Exponential Moving Average (20)",  "Price", "Exponential Moving Average (20)"),
            ("D-Exponential Moving Average (30)",  "Price", "Exponential Moving Average (30)"),
            ("D-Exponential Moving Average (50)",  "Price", "Exponential Moving Average (50)"),
            ("D-Exponential Moving Average (100)", "Price", "Exponential Moving Average (100)"),
            ("D-Exponential Moving Average (200)", "Price", "Exponential Moving Average (200)"),
            ("D-Hull Moving Average (9)", "Price", "Hull Moving Average (9)"),
        ]
        cols_to_print = ["Ticker"]
        for col_out, numerator, denominator in inputs:
            cols_to_print += [col_out]
            if numerator not in cols_to_print:
                cols_to_print += [numerator]
            if denominator not in cols_to_print:
                cols_to_print += [denominator]
        print(cols_to_print)
        for day, df in self.data.items():
            for col_out, numerator, denominator in inputs:
                # print(col_out, numerator, denominator)
                self.data[day][col_out] = df.apply(
                            lambda row: self.volatility_lambda(
                                row[numerator], 
                                row[denominator], 
                                row["Description"]),
                            axis=1)
                # print(self.data[day][col_out])
                # print(self.data[day][[col_out, numerator, denominator]])
        print("Done!")

    # n/d - 1.0
    def volatility_lambda(self, numerator, denominator, description):
        if description is not None and denominator != 0:
            return numerator/denominator - 1.0
        else:
            return 0

    # 5
    ######################################### v8
    # cols
    def calculate_ADX_filtered_RSI(self):
        self.ps()
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

    # 4
    ######################################### v8
    # cols UD through UE, UU through UV
    def calculate_avg_pr(self):
        self.ps()
        print("Calculating average of Percent Rank for the following columns:")
        inputs = [  # col_out, a, b
            ("Avg-# VD", "PR-Avg VD", "PR-# VD"),
            ("Avg-# VU", "PR-Avg VU", "PR-# VU"),
        ]
        cols_to_print = ["Ticker"]
        for col_out, a, b in inputs:
            cols_to_print += [col_out, a, b]
        print(cols_to_print)
        for day, df in self.data.items():
            for col_out, a, b in inputs:
                self.data[day][col_out] = df.apply(
                    lambda row: self.avg_pr_lambda(
                        row[a], 
                        row[b], 
                        description=row["Description"]),
                    axis=1
                )
                # print(self.data[day][[col_out, a, b]])
        print("Done!")        
        
    def avg_pr_lambda(self, *cols, description):
        count = 0
        sum = 0
        if description is not None:
            for col in cols:
                if col > 0:
                    count += 1.0
                    sum += col
            if count > 0:
                return sum/count
            else:
                return 0
        else:
            return 0

    # 3
    ######################################### v8
    # cols UW through UX, UF through UG
    def calculate_percent_rank_avg_volume(self):
        self.ps()
        print("Calculating Percent Rank for the following columns:")
        inputs = [  # col_out, col
            ("PR-Avg VD", "Avg VD"),
            ("PR-Avg VU", "Avg VU"),
            ("PR-# VU", "# VU"),
            ("PR-# VD", "# VD"),
        ]
        cols_to_print = ["Ticker"]
        for col_out, col in inputs:
            cols_to_print += [col_out, col]
        print(cols_to_print)
        for day, df in self.data.items():
            for col_out, col in inputs:
                temp = df[col].replace(0, np.nan)
                self.data[day][col_out] = temp.rank(pct=True).replace(np.nan, 0)
                # self.data[day][col_out] = df[col].rank(pct=True)
                # print(self.data[day][[col_out, col]])
        print("Done!")        

    # 2
    ######################################### v8
    # cols UD through UI and UU through UZ
    def calculate_avg_volume(self):
        self.ps()
        print("Calculating avg volume for the following columns:")
        inputs = [ # col_out_l, col_out_r, a, b, c, d, e, f, g, h, i, j, k
            ("# VD", "Avg VD", "D-Relative Volume VD", "Volume 60/90 VD", "Volume 30/90 VD", "Volume 30/60 VD", "Volume 10/90 VD", "Volume 10/30 VD", "Volume/90 VD", "Volume/60 VD", "Volume/30 VD", "Volume/10 VD", "Volume/10 VD"),
            ("# VU", "Avg VU", "D-Relative Volume VU", "Volume 60/90 VU", "Volume 30/90 VU", "Volume 30/60 VU", "Volume 10/90 VU", "Volume 10/30 VU", "Volume/90 VU", "Volume/60 VU", "Volume/30 VU", "Volume/10 VU", "Volume/10 VU")
        ]
        cols_to_print = ["Ticker"]
        for col_out_l, col_out_r, a, b, c, d, e, f, g, h, i, j, k in inputs:
            cols_to_print += [col_out_l, col_out_r, a, b, c, d, e, f, g, h, i, j, k]
        print(cols_to_print)
        for day, df in self.data.items():
            # cols_to_print = ["Ticker"]
            # Calculate 
            for col_out_l, col_out_r, a, b, c, d, e, f, g, h, i, j, k in inputs:
                self.data[day][[col_out_l, col_out_r]] = df.apply(
                    lambda row: self.calculate_avg_volume_lambda(
                        row[a], 
                        row[b], 
                        row[c], 
                        row[d], 
                        row[e], 
                        row[f], 
                        row[g], 
                        row[h], 
                        row[i], 
                        row[j], 
                        row[k], 
                        # row[l], 
                        description=row["Description"]),
                    axis=1, result_type="expand")
                # self.data[day][percentile_rank_col] = self.data[day][col_out_name].rank(pct=True)
            # print(self.data[day][cols_to_print])
        print("Done!")

    def calculate_avg_volume_lambda(self, *cols, description):
        count = 0
        sum = 0
        # cols = [a, b, c, d, e, f, g, h, i, j, k]
        if description is not None:
            for col in cols:
                if col != 0:
                    count += 1.0
                    sum += col
            if count != 0:
                return count, sum/count
            else:
                return count, 0     
        else:
            return 0, 0

    # 1
    ######################################### v8
    # cols UJ through UT and VA through VL
    def calculate_d_relative_volume_up_or_down(self):
        self.ps()
        print("Calculating d-relative volume up and down cols")
        inputs = [ # col_out, col
            ("Volume 60/90 VD", "Volume 60/90"),
            ("Volume 30/90 VD", "Volume 30/90"),
            ("Volume 30/60 VD", "Volume 30/60"),
            ("Volume 10/90 VD", "Volume 10/90"),
            ("Volume 10/30 VD", "Volume 10/30"),
            ("Volume/90 VD", "Volume/90"),
            ("Volume/60 VD", "Volume/90"),
            ("Volume/30 VD", "Volume/90"),
            ("Volume/10 VD", "Volume/90"),    
            ("Volume/10 VD", "Volume/90"),
            ("D-Relative Volume VD", "D-Relative Volume"),
            ("Volume 60/90 VU", "Volume 60/90"),
            ("Volume 30/90 VU", "Volume 30/90"),
            ("Volume 30/60 VU", "Volume 30/60"),
            ("Volume 10/90 VU", "Volume 10/90"),
            ("Volume 10/30 VU", "Volume 10/30"),
            ("Volume/90 VU", "Volume/90"),
            ("Volume/60 VU", "Volume/90"),
            ("Volume/30 VU", "Volume/90"),
            ("Volume/10 VU", "Volume/90"),    
            ("Volume/10 VU", "Volume/90"),
            ("D-Relative Volume VU", "D-Relative Volume"),    
        ]

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
            for col_out, col in inputs:
                if "VU" in col_out:
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

    # 0
    ######################################### v8
    # cols vm through vv
    def calculate_volume_div(self):
        self.ps()
        print("Calculating the following columns...")
        inputs = [ # col_out, numerator, denominator
                ("Volume 60/90", "Average Volume (60 day)", "Average Volume (90 day)"),
                ("Volume 30/90", "Average Volume (30 day)", "Average Volume (90 day)"),
                ("Volume 30/60", "Average Volume (30 day)", "Average Volume (60 day)"),
                ("Volume 10/90", "Average Volume (10 day)", "Average Volume (90 day)"),
                ("Volume 10/30", "Average Volume (10 day)", "Average Volume (30 day)"),
                ("Volume/90", "Volume", "Average Volume (90 day)"),
                ("Volume/60", "Volume", "Average Volume (60 day)"),
                ("Volume/30", "Volume", "Average Volume (30 day)"),
                ("Volume/10", "Volume", "Average Volume (10 day)"),
                
            ]
        cols_to_print = []
        for col_out, numerator, denominator in inputs:
            cols_to_print += [col_out]
        print(cols_to_print)
        for day, df in self.data.items():
            for col_out, numerator, denominator in inputs:
                self.data[day][col_out] = df.apply(
                            lambda row: self.volatility_lambda(row[numerator], row[denominator], row["Description"]),
                            axis=1)
            # print(self.data[day][cols_to_print])
        print("Done!")
    # End def calculate_volume_div


    def relative_volume_div_lambda(self, relative_volume, div, description):
        return relative_volume/div - 1.0 if description is not None else 0
        
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
        return relative_volume/div - 1.0 if description is not None else 0
        
    def print_data_summary(self):
        for name, day in self.data_dict.items():
            print("==========================================")
            print(name)
            for label, data in day.items():
                print(data.head())
        print("==========================================\n\n\n\n\n")
        print(self.data_dict)

    def export_calculated_data(self):
        print("Exporting all calculated columns to csv by day...")
        cols_to_print = [
            "Ticker",
            "Hull Moving Average (9)",
            "Ichimoku Leading Span A (9, 26, 52, 26)",
            "Ichimoku Leading Span B (9, 26, 52, 26)",
            "Parabolic SAR",
            "Exponential Moving Average (5)",
            "Exponential Moving Average (10)",
            "Exponential Moving Average (20)",
            "Exponential Moving Average (30)",
            "Exponential Moving Average (50)",
            "Exponential Moving Average (100)",
            "Exponential Moving Average (200)",
            "D-Hull Moving Average (9)",
            "Ichimoku Span A/B-1",
            "D-Ichimoku Leading Span A (9, 26, 52, 26)",
            "D-Ichimoku Leading Span B (9, 26, 52, 26)",
            "D-Ichimoku Leading Span A/B",
            "D-Parabolic SAR",
            "D-Exponential Moving Average (5)",
            "D-Exponential Moving Average (10)",
            "D-Exponential Moving Average (20)",
            "D-Exponential Moving Average (30)",
            "D-Exponential Moving Average (50)",
            "D-Exponential Moving Average (100)",
            "D-Exponential Moving Average (200)",
            "Fast EMA Avg",
            "Slow EMA Avg",
            "EMA Gap Slow-Fast",
            "D-Exponential Moving Average (5/10)",
            "D-Exponential Moving Average (10/20)",
            "D-Exponential Moving Average (20/30)",
            "D-Exponential Moving Average (20/50)",
            "D-Exponential Moving Average (50/100)",
            "D-Exponential Moving Average (100/200)",
            "Hull MA Trend U",
            "Ichimoku Trend U",
            "Parabolic Trend U",
            "EMA (5) Trend U",
            "EMA (10) Trend U",
            "EMA (20) Trend U",
            "EMA (30) Trend U",
            "EMA (50) Trend U",
            # "EMA (100 Trend U",
            "EMA (200) Trend U",
            "EMA Avg Trend U",
            "MACD L>S Mom Up",
            "CMF Trend U",
            "MFI Trend U",
            "Ichimoku Trend D",
            "Hull MA Trend D",
            "Parabolic Trend D",
            "EMA (5) Trend D",
            "EMA (10) Trend D",
            "EMA (20) Trend D",
            "EMA (30) Trend D",
            "EMA (50) Trend D",
            "EMA (100) Trend D",
            "EMA (200) Trend D",
            "EMA Avg Trend D",
            "CMF Trend D",
            "MFI Trend D",
            # "Aroon Down (14)",
            # "Aroon Up (14)",
            "Average Directional Index (14)",
            # "Positive Directional Indicator (14)",
            # "Negative Directional Indicator (14)",
            "Awesome Oscillator",
            "Bull Bear Power",
            "Commodity Channel Index (20)",
            "Momentum (10)",
            "MACD Level (12, 26)",
            "MACD Signal (12, 26)",
            "Rate Of Change (9)",
            "Relative Strength Index (7)",
            "Relative Strength Index (14)",
            "Stochastic RSI Fast (3, 3, 14, 14)",
            "Stochastic RSI Slow (3, 3, 14, 14)",
            "Stochastic %K (14, 3, 3)",
            "Stochastic %D (14, 3, 3)",
            "Williams Percent Range (14)",
            "Aroon Up-Down",
            "D+/ADX",
            "D-/ADX",
            "Positive/Negative Directional Indicator (14)",
            # "Aroon Mom Up",
            # "ADX Mom Up",
            # "AO Mom Up",
            # "CCI Mom Up",
            # "RSI (7) 50-70 Mom Up",
            # "RSI (14) 50-70 Mom Up",
            # "RSI 7/14 Mom Up",
            # "MACD L Mom Up",
            # "Aroon Mom Down",
            # "ADX Mom Down",
            # "AO Mom Down",
            # "CCI Mom Down",
            "RSI (14) 30-50 Mom Down",
            "RSI (7) 30-50 Mom Down",
            "RSI 7/14 Mom Down",
            "MACD L Mom Down",
            "MACD L<S Mom Down",
            "Bollinger Price>Upper",
            "Keltner CH Price>Upper",
            # "RSI (7) Overbought",
            # "RSI (14) Overbought",
            "Stoch K% Overbought",
            "Stoch D% Overbought",
            "Willams% Overbought",
            "CCI Overbought",
            "CMF Overbought",
            "MFI Overbought",
            "Relative Volume Overbought",
            "Bollinger Price<Lower",
            "Keltner CH Price<Lower",
            "RSI (7) Oversold",
            "RSI (14) Oversold",
            "Stoch K% Oversold",
            "Stoch D% Oversold",
            "Williams% Oversold",
            "CCI Oversold",
            "CMF Oversold",
            "MFI Oversold",
            "Reative Volume Oversold",
            "MACD L Signal",
            "MACD S Signal",
            "AO Signal",
            "CCI Signal",
            "D-Ichimoku Leading Span A/B Signal",
            "Bollinger Upper/Lower Band (20) Signal",
            "Donchian CH Price/Lower-1 Signal",
            "Donchian CH  (Price-Upper)/Upper Signal",
            "Keltner CH Upper/Lower Band (20) Signal",
            "Hull MA Signal",
            "D-Pivot Fibonacci R3",
            "D-Pivot Fibonacci R2",
            "D-Pivot Fibonacci R1",
            "D-Pivot Fibonacci P",
            "D-Pivot Fibonacci S1",
            "D-Pivot Fibonacci S2",
            "D-Pivot Fibonacci S3",
            "Fibonacci Minimum",
            "Closest D-Pivot",
            "Current Pivot Level",
            "ATR/Price",
            "ADR/Price",
            "ADR/ATR (Price)",
            "Price/VWAP",
            "Price/VWMA (20)",
            "D-1-Month Low",
            "D-3-Month Low",
            "D-6-Month Low",
            "D-52 Week Low",
            "D-1-Month High",
            "D-3-Month High",
            "D-6-Month High",
            "D-52 Week High",
            "D-1-Month Low/ATR",
            "D-3-Month Low/ATR",
            "D-6-Month Low/ATR",
            "D-52 Week Low/ATR",
            "D-1-Month High/ATR",
            "D-3-Month High/ATR",
            "D-6-Month High/ATR",
            "D-52 Week High/ATR",
            "Ultimate Oscillator (7, 14, 28)",
            "UO Overbought",
            "UO Trend U",
            "UO Trend D",
            "UO Oversold",
            "Stochastic RSI Fast/Slow (3, 3, 14, 14)",
            "Stochastic %K/%D (14, 3, 3)",
            "ADX Filtered RSI (14)",
            "ADX Filtered RSI (7)",
            "ADX Filtered RSI (7/14)",
            "Relative Strength Index (7/14)",
            "Volatility Month",
            "Volatility Week",
            "Volatility",
            "1-Year Beta",
            "Bollinger Lower Band (20)",
            "Bollinger Upper Band (20)",
            "Donchian Channels Lower Band (20)",
            "Donchian Channels Upper Band (20)",
            "Keltner Channels Lower Band (20)",
            "Keltner Channels Upper Band (20)",
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
            "Donchian CH Price/Upper-1",
            "Chaikin Money Flow (20)",
            "Money Flow (14)",
            "Relative Volume",
            "Volume",
            "Average Volume (10 day)",
            "Average Volume (30 day)",
            "Average Volume (60 day)",
            "Average Volume (90 day)",
            "D-Relative Volume",
            "Relative Volume/10",
            "Relative Volume/30",
            "Relative Volume/60",
            "Relative Volume/90",
            "Relative Volume 10/30",
            "D-Relative Volume-Up",
            "Relative Volume/10-Up",
            "Relative Volume/30-Up",
            "Relative Volume/60-Up",
            "Relative Volume/90-Up",
            "D-Relative Volume-Down",
            "Relative Volume/10-Down",
            "Relative Volume/30-Down",
            "Relative Volume/60-Down",
            "Relative Volume/90-Down",
            ]
        out_path = "data_out/"
        for day, df in self.data.items():
            # df[cols_to_print].to_csv(out_path + day + ".csv")
            df.to_csv(out_path + day + ".csv")
            
        print("Done!")
            
        

if __name__ == "__main__":
    try:
        model = Model()
    except Exception as e:
        print(e)

            
        