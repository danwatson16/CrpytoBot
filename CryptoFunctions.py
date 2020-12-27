import numpy as np
import cbpro
import datetime as datetime
from datetime import timedelta
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import pandas_ta as ta
import statistics
import operator
import time





def add_cols(new_df, period=20):
    new_df["SMA"] = new_df["close"].rolling(window=period).mean()
    new_df["STD"] = new_df["close"].rolling(window=period).std()
    new_df["Upper"] = new_df["SMA"] + (new_df["STD"] * 2)
    new_df["Lower"] = new_df["SMA"] - (new_df["STD"] * 2)
    return new_df


def get_signal(data):
    buy_signal = []
    sell_signal = []
    for i in range(len(data["close"])):
        if data["close"][i] > data["Upper"][i]:
            buy_signal.append(np.nan)
            sell_signal.append(data["close"][i])
        elif data["close"][i] < data["Lower"][i]:
            buy_signal.append(data["close"][i])
            sell_signal.append(np.nan)
        else:
            buy_signal.append(np.nan)
            sell_signal.append(np.nan)
    return (buy_signal, sell_signal)


def get_new(new_df):
    new_df["Bollinger Buy"] = get_signal(new_df)[0]
    new_df["Bollinger Sell"] = get_signal(new_df)[1]
    return new_df


def bollinger_plot(company_name, new_df, time=0):
    fig, ax = plt.subplots(4, figsize=(15, 15))
    x_axis = new_df.index[time:]
    ax[0].fill_between(x_axis, new_df["Upper"][time:], new_df["Lower"][time:], color="grey")
    ax[0].plot(x_axis, new_df["close"][time:], color="Gold", lw=2, label="Close price")
    ax[0].plot(x_axis, new_df["SMA"][time:], color="Blue", lw=2, label="SMA")
    ax[1].plot((ta.coppock(new_df.close)))
    (buy_signal, sell_signal, buy_index, sell_index) = get_coppock(new_df)
    ax[1].scatter(buy_index, buy_signal, color="green", label="Buy signal", marker="^", alpha=1)
    ax[1].scatter(sell_index, sell_signal, color="red", label="Sell signal", marker="v", alpha=1)
    MACD = ta.macd(new_df.close, fast = 5, slow = 35, signal = 5)
    buy_signal, sell_signal, buy_index, sell_index = get_MACD(new_df, fast = 5, slow = 35, signal = 5)
    ax[2].bar(MACD.iloc[:,1].index, MACD.iloc[:,1], label="bar")
    ax[2].plot(MACD.iloc[:,0] , label="MACD")
    ax[2].plot(MACD.iloc[:,2], label="MACDS")
    ax[2].scatter(buy_index, buy_signal, color="red", label="sell signal", marker="v", alpha=1)
    ax[2].scatter(sell_index, sell_signal, color="green", label="buy signal", marker="^", alpha=1)
    ax[2].legend()
    ax[1].legend()
    ax[0].scatter(x_axis, new_df["Bollinger Buy"][time:], color="green", label="Buy signal", marker="^", alpha=1)
    ax[0].scatter(x_axis, new_df["Bollinger Sell"][time:], color="red", label="Sell signal", marker="v", alpha=1)
    ax[0].set_title(company_name)
    ax[3].plot(ta.rsi(new_df.close))
    ax[3].axhline(30, linestyle="--", color="red")
    ax[3].axhline(70, linestyle="--", color="green")
    plt.legend()
    plt.savefig("{}".format(str(datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")) + ".png"))


def get_MACD(df, fast = 12, slow = 26, signal = 9):
    MACD = ta.macd(df.close, fast = fast, slow = slow, signal = signal)
   # print(MACD.iloc[:,2])
    crosses = np.argwhere(np.diff(np.sign(MACD.iloc[:,0] - MACD.iloc[:,2]))).flatten()
    buy_signal = []
    sell_signal = []
    buy_index = []
    sell_index = []
    values = []
    values2 = []
    for i in (MACD.iloc[:,0][crosses]):
        values.append(i)
    for i in (MACD.iloc[:,2][crosses]):
        values2.append(i)
    # print(values, values2)
    for i in range(len(values)):
        if values[i] > values2[i]:
            buy_index.append(MACD.iloc[:,0][crosses].index[i])
            buy_signal.append(values[i])
        elif values[i] <= values2[i]:
            sell_index.append(MACD.iloc[:,2][crosses].index[i])
            sell_signal.append(values2[i])
    return (buy_signal, sell_signal, buy_index, sell_index)


def get_coppock(df):
    coppock = ta.coppock(df.close)
    small = abs(ta.coppock(df.close).diff()) < 0.3
    second = ta.coppock(df.close).diff().diff()
    buy_signal = []
    sell_signal = []
    buy_index = []
    sell_index = []
    for i in range(len(small)):
        #  print(small[i], i)
        if small[i]:
            #   print(second[i])
            if np.sign(second[i]) == 1:
                sell_index.append(i)
                sell_signal.append(coppock[i])
            else:
                buy_index.append(i)
                buy_signal.append(coppock[i])
    return (buy_signal, sell_signal, buy_index, sell_index)

def get_dataframe_days(start, currency, end, granularity = 86400):
    adict = {}
    historicData = auth_client.get_product_historic_rates(currency, start = start, end = end, granularity = granularity)
    print(historicData)
    historicData.reverse()
    for val in range(len(historicData)):
        day = str(datetime.datetime.strptime(start, '%Y-%m-%d') + timedelta(days = val))[:10]
        adict[day] = historicData[val]
    df = pd.DataFrame.from_dict(adict, orient = "index")
    df = df.rename(columns = {0 : "time", 1 : "low", 2 : "high", 3 : "open", 4 : "close", 5 : "volume"})
    return df

def getSpecificAccount(cur):
    x = auth_client.get_accounts()
    for account in x:
        if account['currency'] == cur:
            return account['id']
currency = 'ETH-GBP'
currency_id = getSpecificAccount(currency[:3])

def get_RSI(new_data, funds):
    RSI = ta.rsi(new_data.close)
    current_rsi = RSI.iloc[-1]
    RSI_gradient = RSI.diff().iloc[-1]
    mean_rsi = np.nanmean(RSI[-1-100:-1])
    RSI_difference = current_rsi - mean_rsi
    return (RSI, current_rsi, RSI_gradient, mean_rsi, RSI_difference)

def get_Kelly(win, loss, gain_list, loss_list):
    win_prob = win / (win + loss)
    wl = statistics.mean(gain_list) / abs(statistics.mean(loss_list))
   # print(win_prob)
    #print(wl)
    kelly = win_prob - (1 - win_prob) / wl
    return kelly

def update(difference, win, loss, loss_list, gain_list, held):
    if difference > 0:
        win += 1
        gain_list.append(difference * held)
    else:
        loss += 1
        loss_list.append(difference * held)
    if len(loss_list) == 0:
        loss_list.append(-0.001)
    elif len(gain_list) == 0:
        gain_list.append(0.001)
    return (difference, win, loss, loss_list, gain_list, held)

def get_sharpe(total_profits):
    length = (len(total_profits))
    R = pd.DataFrame(total_profits).cumsum()
    r = (R - R.shift(1)) / R.shift(1)
    sharpe = r.mean() / r.std() * np.sqrt(length)
    return sharpe[0]

def get_balances(k):
    balances = k.get_account_balance().reset_index()
    try:
        funds = balances[balances["index"] == "ZGBP"]["vol"][0]
    except:
        funds = 0
        pass
    try:
        amount_held = balances[balances["index"] == "XETH"]["vol"][1]
    except:
        amount_held = 0
        pass
    return (funds, amount_held)

def amount_deposited(k):
    deposits = k.get_ledgers_info(type="deposit")[0]
    total_deposited = 0
    for i in range(len(deposits)):
        price_at_deposit = k.get_ohlc_data("ETHGBP", since=deposits["time"][i] - 1000, ascending=True, interval=15)[0]["close"][0]
        total_deposited += deposits["amount"][i] * price_at_deposit
    return total_deposited

def overall_profit(k):
    fees = k.get_ledgers_info()[0]["fee"].sum()
    total_deposited = amount_deposited(k)
    current_price = k.get_ohlc_data("ETHGBP")[0]["close"][0]
    (funds, amount_held) = get_balances(k)
    overall_profit = (current_price * amount_held + funds) - total_deposited - fees
    return overall_profit

def previous_win_loss(k):
    ledgers = k.get_ledgers_info()[0]
    ledgers["price"] = get_prices(k)
    balances = []
    for val in range(0, len(ledgers), 2):
        mini_ledgers = ledgers[val:val + 2]
        profit = 0
        for i in range(len(mini_ledgers)):
            if mini_ledgers["asset"][i] == "XETH":
                profit += mini_ledgers["balance"][i] * mini_ledgers["price"][i]
            else:
                profit += mini_ledgers["balance"][i]
        balances.append(profit)
    balances = pd.DataFrame(balances)
    differences = balances.diff()
    win = 0
    loss = 0
    gain_list = []
    loss_list = []
    for val in differences.iloc[:, 0]:
        if val > 0:
            win += 1
            gain_list.append(val)

        elif val <= 0:
            loss += 1
            loss_list.append(val)
    print("WIn", win)
    print("Loss", loss)
    print("Gain list", gain_list)
    print("LOss list", loss_list)
    kelly = get_Kelly(win, loss, gain_list, loss_list)
    return kelly, list(differences), win, loss, gain_list, loss_list

def running_sharpe(k):
    kelly, differences, win, loss, gain_list, loss_list = previous_win_loss(k)
    return get_sharpe(total_profits= differences)

def previously_bought(k):
    orders = k.get_closed_orders()[0]
    all_orders = orders[orders["reason"].isnull()]
    buy = False
    if (all_orders.iloc[0,]["descr_type"] == "buy"):
        buy = True
    return buy

def get_prices(k):
    ledgers = k.get_ledgers_info()[0]
    trades = k.get_trades_history()[0]
    prices = []
    for val in range(0, len(ledgers), 2):
        # print(ledgers["time"][val])
        test = k.get_ohlc_data("ETHGBP", since=ledgers["time"][val] - 1000, ascending=True, interval=5)[0]
        time.sleep(0.1)
        date = trades.index[val]
        #print(date)
        #print(date)

        dates = []
        for i in range(len(test)):
            dates.append(abs(date - test.index[i]))
            min_index, min_time = min(enumerate(dates), key=operator.itemgetter(1))
        prices.append(test["close"][min_index])
        time.sleep(5)
        new_prices = []
    for i in range(len(prices)):
        new_prices.append(prices[i])
        new_prices.append(prices[i])
    return new_prices

def get_buy_sell_percentage(k, hours = 1):
    test = pd.Timestamp(datetime.datetime.now() - datetime.timedelta(hours=hours))
    trades = k.get_recent_trades("ETHGBP")[0]
    min_index, min_time = min(enumerate(abs(trades.index - test)), key=operator.itemgetter(1))
    new_trades = trades[:min_index]
    number_buy = len(new_trades[new_trades["buy_sell"] == "buy"])
    number_sell = len(new_trades[new_trades["buy_sell"] == "sell"])
    buy_vol = new_trades[new_trades["buy_sell"] == "buy"]["volume"].sum()
    buy_vol = new_trades[new_trades["buy_sell"] == "sell"]["volume"].sum()
    sell_percentage = number_sell / len(new_trades)
    buy_percentage = number_buy / len(new_trades)
    return buy_percentage, sell_percentage





