import CryptoFunctions
import pandas_ta as ta
import numpy as np
import datetime
import krakenex
from pykrakenapi import KrakenAPI
import pandas as pd
import cbpro
import time
import statistics

api = krakenex.API(key = "27cgHwEM02L02J8PtEwMV4V/+Padjduq2Iu7p+yPaO+cq5oqWshakW+c", secret = "DnB7Ny8pZjySgr82Ug1bIWv/12NLBFObiagX3lLVCLT1/kqBo4XIHzf+a5lVPpAYBMHG2N2XNlKx3lgTLD4BqA==")
k = KrakenAPI(api)

apiKey = "1e0aab54a5c0b8d29965c7a4f42e50c4"
apiSecret = "XNcfaRxfoJMJ/8Hf88qG7enhR8af95b5AVSW5+twhOrX/NWMsMwdZCG6GGsr0hdmu8+Wc2lfq8LOIUfVxPU9Ig=="
passphrase = "x6dnvvb2ge"

auth_client = cbpro.AuthenticatedClient(apiKey,apiSecret,passphrase)
funds = 10

def MACD_backtest(df, funds = 10, difference_coef = 2, current_coef = 1e01, intercept = funds/2, win = 0, loss = 0,
    gradient = True, gradient_threshold = 0.1, difference_bool = True, sell_params = True, difference_threshold = 10, fast = 12, slow = 26, signal = 9,
    gain_list = [], loss_list = [], kelly = 0.5, RSI_bool = True, RSI_buy_parameter = 40, RSI_sell_parameter = 60):
    original_funds = funds
    RSI = ta.rsi(df.close)
    funds = funds
    (buy_signal, sell_signal, sell_index, buy_index) = CryptoFunctions.get_MACD(df, fast = fast, slow = slow, signal = signal)
    MACD = ta.macd(df.close)
    if sell_index[0] < buy_index[0]:
        del sell_index[0]
    if buy_index[-1] > sell_index[-1]:
        del buy_index[-1]
    total_profit = 0
    total_exposure = 0
    length = len(buy_index)
    for index in range(len(buy_index)):
        current_rsi = RSI[buy_index[index]]
        if buy_index[index] <= 50:
            mean_rsi = np.nanmean(RSI[buy_index[index] - 20:buy_index[index]])
        else:
            mean_rsi = np.nanmean(RSI[buy_index[index]-50:buy_index[index]])
        RSI_difference = current_rsi - mean_rsi
        if gradient and difference_bool and RSI_bool:
            #print("COP")
            macd_buy_gradient = MACD.iloc[:,0].diff()[buy_index[index]]
            macd_sell_gradient = MACD.iloc[:,0].diff()[sell_index[index]]
            #if macd_buy_gradient <= gradient_threshold and RSI_difference >= -difference_threshold and
            if current_rsi > RSI_buy_parameter:
                if index + 1 < length:
                    index += 1
            if sell_params:
                #if macd_sell_gradient >= -gradient_threshold and RSI_difference <= difference_threshold and \
                if current_rsi < RSI_sell_parameter:
                    if index + 1 < length:
                        index += 1
        #print(kelly)
        amount_to_invest = kelly*funds
        total_exposure += amount_to_invest
        funds -= amount_to_invest
        buy_price = df.iloc[buy_index[index],4]
        held = amount_to_invest/buy_price
        sell_price = df.iloc[sell_index[index],4]
        funds += sell_price*held
        difference = sell_price - buy_price
        if difference >= 0:
            win += 1
            gain_list.append(difference*held)
        else:
            loss += 1
            loss_list.append(difference*held)
        total_profit += held*difference
        fees = 0.0025*amount_to_invest + 0.0025*sell_price*held
        total_profit -= fees
    gain_percentage = (total_profit/original_funds)*100
    return (total_profit, total_exposure, total_exposure/gain_percentage, gain_percentage, win, loss, gain_list, loss_list)


def total_MACD_backtest(funds=10, periods=100, difference_coef=2, current_coef=1e02, intercept=funds / 2):
    alist = np.arange(0.0, 0.10, 0.01)
    adict_12269 = {}
    adict_5355 = {}
    alist_2 = [i for i in range(0, 10, 1)]
    macd_list = [(12,26,9), (5,35,5)]
    for val in macd_list:
        fast = val[0]
        slow = val[1]
        signal = val[2]
        for k in alist_2:
            for j in alist:
                win = 0
                loss = 0
                print("Gradient threshold", j, "Difference threshold", k)
                print("MACD", fast, slow, signal)
                total_profits = []
                gain_percentages = []
                total_exposures = []
                exposure_gain_ratios = []
                funds = funds
                for i in range(1, periods, 3):
                    start = str(datetime.datetime.now().date() - datetime.timedelta(days=2) - datetime.timedelta(days=i))
                    end = str(datetime.datetime.now().date() + datetime.timedelta(days=1) - datetime.timedelta(days=i))
                    try:
                        new_data = auth_client.get_product_historic_rates("ETH-GBP", start=start, end=end, granularity=900)
                        new_data.reverse()
                      #  print(new_data)
                    except:
                        print(new_data)

                    new_data = pd.DataFrame(new_data)
                    new_data = new_data.rename(columns={0: "time", 1: "low", 2: "high", 3: "open", 4: "close", 5: "volume"})
                    (total_profit, total_exposure, exposure_gain_ratio, gain_percentage, win, loss, gain_list, loss_list) = MACD_backtest(new_data, funds=funds,
                                                                                                         difference_coef=difference_coef,
                                                                                                         current_coef=current_coef,
                                                                                                         intercept=intercept, gradient = False,  difference_bool= False,
                                                                                                         gradient_threshold= j, difference_threshold= k, fast = fast, slow = slow, signal = signal,
                                                                                                         win = win, loss = loss, gain_list = gain_list, loss_list = loss_list, sell_params =False, RSI_bool=True)
                    total_profits.append(total_profit)
                    total_exposures.append(total_exposure)
                    exposure_gain_ratios.append(exposure_gain_ratio)
                    gain_percentages.append(gain_percentage)
                    time.sleep(0.1)
                print("Gain list", gain_list)
                print("Loss list", loss_list)
                print("Wins", win)
                print("Losses", loss)
                print(win/(win+loss))
                length = (len(total_profits))
                R = pd.DataFrame(total_profits).cumsum()
                r = (R - R.shift(1)) / R.shift(1)
                sharpe = r.mean() / r.std() * np.sqrt(length)
                print("Sharpe ratio", sharpe[0])
                print("Total percentage increase", str((sum(total_profits)/funds)*100) + "%")
                print("Average profit", sum(total_profits)/len(total_profits))
                print("Average gain percentage", sum(gain_percentages)/len(gain_percentages))
                print("Average exposure", sum(total_exposures)/len(total_exposures))
                print("Average exposure-gain ratio", sum(exposure_gain_ratios)/len(exposure_gain_ratios))
                print("\n")
                if fast == 12 and slow == 26 and signal == 9:
                    adict_12269[((fast, slow, signal), j,k)] = [sharpe[0], (sum(total_profits)/funds)*100]
                else:
                    adict_5355[((fast, slow, signal), j,k)] = [sharpe[0], (sum(total_profits)/funds)*100]
    print("12269 dict", adict_12269)
    print("5355 dict", adict_5355)
    return (adict_12269, adict_5355)
    #return (sum(exposure_gain_ratios) / len(exposure_gain_ratios), sum(gain_percentages) / len(gain_percentages),
   #             sum(total_exposures) / len(total_exposures))

#(adict_12269, adict_5355) = total_MACD_backtest()

# df = pd.DataFrame.from_dict(adict_12269, orient = "index").reset_index().rename(columns = {0 : "vals"})
# df["vals"] = df["vals"].astype(float)
# max_value = df["vals"].max()
# max_params = df[df["vals"] == df["vals"].max()]["index"].item()
# print(max_params, max_value)

#Best params are ((5, 35, 5), 0.04, 7) giving sharpe of 3.512657731833407 and ROI of 112.103936

def MACD_backtest_tuned(funds=10, periods=100, difference_coef=2, current_coef=1e02, intercept=funds / 2):
    gain_list = []
    loss_list = []
    win = 0
    loss = 0
    kelly = 0.5
    total_profits = []
    gain_percentages = []
    total_exposures = []
    exposure_gain_ratios = []
    funds = funds
    for i in range(1, periods, 3):
        start = str(datetime.datetime.now().date() - datetime.timedelta(days=2) - datetime.timedelta(days=i))
        end = str(datetime.datetime.now().date() + datetime.timedelta(days=1) - datetime.timedelta(days=i))
        try:
            new_data = auth_client.get_product_historic_rates("ETH-GBP", start=start, end=end, granularity=900)
            new_data.reverse()
        except:
            print(new_data)

        new_data = pd.DataFrame(new_data)
        new_data = new_data.rename(columns={0: "time", 1: "low", 2: "high", 3: "open", 4: "close", 5: "volume"})
        (total_profit, total_exposure, exposure_gain_ratio, gain_percentage, win, loss, gain_list, loss_list) = MACD_backtest(new_data, funds=funds,
                                                                                             difference_coef=difference_coef,
                                                                                             current_coef=current_coef,
                                                                                             intercept=intercept, gradient = True,  difference_bool= True,
                                                                                             gradient_threshold= 0.04, difference_threshold= 7, fast = 5, slow = 35, signal = 5,
                                                                                             win = win, loss = loss, gain_list = gain_list, loss_list = loss_list, kelly = kelly, sell_params= False, RSI_bool= True, RSI_buy_parameter= 36,
                                                                                             RSI_sell_parameter= 50)
        if len(loss_list) == 0:
            loss_list.append(-0.1)
        kelly = CryptoFunctions.get_Kelly(win=win, loss=loss, gain_list=gain_list, loss_list=loss_list)
        total_profits.append(total_profit)
        total_exposures.append(total_exposure)
        exposure_gain_ratios.append(exposure_gain_ratio)
        gain_percentages.append(gain_percentage)
        time.sleep(0.1)
    sharpe = CryptoFunctions.get_sharpe(total_profits)
    print("Sharpe ratio", sharpe)
    print("Total percentage increase", str((sum(total_profits)/funds)*100) + "%")
    print("Average profit", sum(total_profits)/len(total_profits))
    print("Average gain percentage", sum(gain_percentages)/len(gain_percentages))
    print("Average exposure", sum(total_exposures)/len(total_exposures))
    print("Average exposure-gain ratio", sum(exposure_gain_ratios)/len(exposure_gain_ratios))


MACD_backtest_tuned()

