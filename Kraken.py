import time
import datetime as datetime
import CryptoFunctions
import krakenex
from pykrakenapi import KrakenAPI
import pandas_ta as ta

api = krakenex.API(key = "27cgHwEM02L02J8PtEwMV4V/+Padjduq2Iu7p+yPaO+cq5oqWshakW+c", secret = "DnB7Ny8pZjySgr82Ug1bIWv/12NLBFObiagX3lLVCLT1/kqBo4XIHzf+a5lVPpAYBMHG2N2XNlKx3lgTLD4BqA==")
k = KrakenAPI(api)
sell = False
buy = False
total_profit = CryptoFunctions.overall_profit(k)
print("Total profit", total_profit)
iteration = 0
currency = 'ETH-GBP'
kelly, differences, win, loss, gain_list, loss_list = CryptoFunctions.previous_win_loss(k)
while True:
    bought = CryptoFunctions.previously_bought(k)
    buy_percentage, sell_percentage = CryptoFunctions.get_buy_sell_percentage(k, hours = 1)
    print("Buy percentage", buy_percentage)
    print("Sell percentage", sell_percentage)
    can_buy = not bought
    print("Bought", bought)
    print("Can buy", can_buy)
    running_sharpe = CryptoFunctions.get_sharpe(differences)
    print("Running sharpe", running_sharpe)
    print("Kelly", kelly)
    (funds, amount_held) = CryptoFunctions.get_balances(k)
    print("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -")
    print("Current datetime is", datetime.datetime.now())
    print("Iteration number", iteration)
    new_data, last = k.get_ohlc_data("ETHGBP", interval=5, ascending=True,
                                     since=k.datetime_to_unixtime(datetime.datetime.now() - datetime.timedelta(hours=25)))
    new_data = new_data.reset_index()
    #CryptoFunctions.bollinger_plot("ETH", CryptoFunctions.get_new(CryptoFunctions.add_cols(new_data)), time=0)
    current_price = new_data.close.iloc[-1]
    print("Price from data", current_price)
    minimum_order = current_price * 0.02
    number_can_buy = funds / current_price
    print("Amount in wallet is: {}, Current price is: {}, The total £ available in ETH is: {} and Funds are {}".format(
        amount_held, current_price, number_can_buy, funds))
    (RSI, current_rsi, RSI_gradient, mean_rsi, RSI_difference) = CryptoFunctions.get_RSI(new_data, funds)
    amount_to_invest = kelly*funds
    minimum_order = current_price*0.02
    if amount_to_invest <= minimum_order:
        amount_to_invest = minimum_order + 0.1
    print("Amount to invest", amount_to_invest)
    print("Current RSI", current_rsi)
    print("RSI gradient", RSI_gradient)
    print("Mean RSI", mean_rsi)
    print("RSI difference", RSI_difference)
    (buy_signal, sell_signal, sell_index, buy_index) = CryptoFunctions.get_MACD(new_data, fast = 5, slow = 35, signal = 5)
    MACD = ta.macd(new_data.close)
    macd_gradient = MACD.iloc[:,0].diff().iloc[-1]
    print("MACD gradient", macd_gradient)
    length = len(MACD)
    print(buy_index)
    for i in buy_index:
        if (length - 2 == i or length - 3 == i) and can_buy and current_rsi <= 36:
            print("Buy!")
            buy = True
    for j in sell_index:
        if (length - 2 == j or length - 3 == j) and bought:
            print("Sell!")
            sell = True
    if buy and can_buy:
        CryptoFunctions.bollinger_plot("ETH", CryptoFunctions.get_new(CryptoFunctions.add_cols(new_data)), time=0)
        buy_price = current_price
        held = amount_to_invest / buy_price
        print("Calculated amount held", held)
        bought = True
        can_buy = False
        amount_to_invest_ETH = amount_to_invest / current_price
        print("Amount to invest in £: {} and in ETH: {}".format(amount_to_invest, amount_to_invest_ETH))
        print(k.add_standard_order(pair="ETHGBP", type="buy", ordertype="market", volume=amount_to_invest_ETH,
                                   validate=False))
        (funds, amount_held) = CryptoFunctions.get_balances(k)
        print("Funds", funds, "Amount held", amount_held)
        previous_invest = amount_to_invest
    if sell and bought:
        CryptoFunctions.bollinger_plot("ETH", CryptoFunctions.get_new(CryptoFunctions.add_cols(new_data)), time=0)
        print("Sell signal and bought before, selling")
        sell_price = current_price
        difference = sell_price - buy_price
        difference, win, loss, loss_list, gain_list, held = CryptoFunctions.update(difference, win, loss, loss_list, gain_list, held)
        differences.append(difference)
        running_sharpe = CryptoFunctions.get_sharpe(differences)
        print("New sharpe", running_sharpe)
        kelly = CryptoFunctions.get_Kelly(win=win, loss=loss, gain_list=gain_list, loss_list=loss_list)
        print("Profit from trade", amount_held * difference)
        print("Difference", difference)
        print("Total profit after fees", total_profit)
        print("Held before", amount_held)
        print(k.add_standard_order(pair="ETHGBP", type="sell", ordertype="market", volume=amount_held, validate=False))
        (funds, amount_held) = CryptoFunctions.get_balances(k)
        print("Held after", amount_held)
        print("Funds after", funds)
        print("New kelly", kelly)
        can_buy = True
        bought = False

    if sell == False and buy == False:
        print("No buy or sell signals!")

    # Printing here to make the details easier to read in the terminal

    # Wait for 5 minutes before repeating
    iteration += 1
    buy = False
    sell = False
    time.sleep(240)


#TODO TRY OUT DIFFERENT SELL CONDITIONS - SELLING TOO LATE
#TODO TRY OUT TAKEPROFIT/STOP-LOSS ETC
#TODO DIFFERENT METRICS