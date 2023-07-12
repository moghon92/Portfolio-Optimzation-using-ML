import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates
from matplotlib import gridspec

from util import get_data, plot_data
import indicators as ind


def author():
    """  		  	   		   	 		  		  		    	 		 		   		 		  
    :return: The GT username of the student  		  	   		   	 		  		  		    	 		 		   		 		  
    :rtype: str  		  	   		   	 		  		  		    	 		 		   		 		  
    """  		  	   		   	 		  		  		    	 		 		   		 		  
    return "mghoneim3"  # Change this to your user ID


def execute_trades(indicators_df, symbol, sd, ed, sv):
    '''
    function to compute buy sell amounts from trading signals
    '''
    df = indicators_df.copy()
    df['trade_amount'] = 0

    net_holdings = 0
    for i in range(df.shape[0] - 1):
        if df.iloc[i, 4] == 1:  # long
            for trade_val in [2000, 1000, 0]:
                if net_holdings + trade_val in [-1000, 0, 1000]:
                    df.iloc[i, 5] = trade_val
                    net_holdings += trade_val
                    break
        elif df.iloc[i, 4] == -1:
            for trade_val in [-2000, -1000, 0]:
                if net_holdings + trade_val in [-1000, 0, 1000]:
                    df.iloc[i, 5] = trade_val
                    net_holdings += trade_val
                    break

    out = df.iloc[:, 5]

    return out[:ed].to_frame(name=symbol)


def signals(indicators_df):
    '''
       function to compute buy/sell signals from indicators
    '''
    df = indicators_df.copy()

    # if bollinger band crosses the thresholds 2+ times withing 60 days generate a signal
    df['prev_pB'] = df['pB'].shift(1)
    df['pB_sell_signal'] = df.apply(lambda x: 1 if x.pB < 2 and x.prev_pB > 2 else 0, axis=1)
    df['pB_buy_signal']  = df.apply(lambda x: 1 if x.pB >-2 and x.prev_pB < -2 else 0, axis=1)
    df['rolling_pB_sell_signal'] = df['pB_sell_signal'].rolling(60).sum()
    df['rolling_pB_buy_signal'] = df['pB_buy_signal'].rolling(60).sum()
    df['pB_signal_flag'] = df.apply(lambda x: 'buy' if x['rolling_pB_buy_signal'] >= 2 else 'sell' if x['rolling_pB_sell_signal'] >=2  else np.nan,axis=1)


    # if RSI crosses threshold 1+ times in last 28 says generate signal
    df['prev_RSI'] = df['RSI'].shift(1)
    df['RSI_sell_signal'] = df.apply(lambda x: 1 if x.RSI < 2 and x.prev_RSI > 2 else 0, axis=1)
    df['RSI_buy_signal']  = df.apply(lambda x: 1 if x.RSI > -2 and x.prev_RSI < -2 else 0, axis=1)
    df['rolling_RSI_sell_signal'] = df['RSI_sell_signal'].rolling(28).max()
    df['rolling_RSI_buy_signal'] = df['RSI_buy_signal'].rolling(28).max()
    df['RSI_signal_flag'] = df.apply(lambda x: 'buy' if x['rolling_RSI_buy_signal']==1 else 'sell' if x['rolling_RSI_sell_signal'] == 1 else np.nan, axis=1)

    # if PPO crosses the 0 line generate signal
    df['prev_PPO'] = df['PPO'].shift(1)
    df['PPO_sell_signal'] = df.apply(lambda x: 1 if x.PPO < 0 and x.prev_PPO > 0 else 0, axis=1)
    df['PPO_buy_signal']  = df.apply(lambda x: 1 if x.PPO > 0 and x.prev_PPO < 0 else 0, axis=1)
    df['ppo_signal_flag'] = df.apply(lambda x: 'buy' if x['PPO_buy_signal']==1 else 'sell' if  x['PPO_sell_signal'] == 1 else np.nan, axis=1)
    df['ppo_signal_flag'] = df['ppo_signal_flag'].ffill(axis=0)


    #df['prev_EMV'] = df['EMV'].shift(1)


    # combine all signals into 1 startegy
    # iff PPO signal AND (RSI or %B signal) the buy/sell
    df['signal'] = df.apply(lambda row: 1 if (row['PPO_buy_signal'] == 1) and (
                row['RSI_signal_flag'] == 'buy' or row['pB_signal_flag'] == 'buy') \
        else -1 if (row['PPO_sell_signal'] == 1)  and (
                row['RSI_signal_flag'] == 'sell' or row['pB_signal_flag'] == 'sell') else 0, axis=1)
    return df['signal']



def testPolicy(symbol='JPM', sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009,12,31), sv = 100000, verbose=False):
    '''
    test manual startegy. reads prices data, generates indicators and combines them into a strategy
    '''
    tickers = [symbol]
    try:
        price = get_data(tickers, pd.date_range(sd, ed))
    except:
        return 'sorry symbol or date ranges does not exist'

    price = price[tickers]

    RSI = ind.Rel_SI(price, lookback=14)
    RSI.rename(columns={symbol:'RSI'}, inplace=True)
    pB = ind.Bollinger_bands(price, lookback=14)
    pB.rename(columns={symbol:'pB'}, inplace=True)
    ppo = ind.prec_price_oscillator(price)
    ppo.rename(columns={symbol:'PPO'}, inplace=True)
    #emv = ind.EMV(price)
    #emv.rename(columns={symbol: 'EMV'}, inplace=True)


    indicators_df = price.join(RSI).join(pB).join(ppo)#.join(emv)

    indicators_df['signal'] = signals(indicators_df)

    trades = execute_trades(indicators_df, symbol, sd, ed, sv)

    return trades

def testPolicy_with_plot(symbol='JPM', sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009,12,31), sv = 100000):
    '''
    same as previous function but with added option o plot a chart for report
    '''

    tickers = [symbol]
    try:
        price = get_data(tickers, pd.date_range(sd, ed))
    except:
        return 'sorry symbol or date ranges does not exist'
    price = price[tickers]

    RSI = ind.Rel_SI(price, lookback=14)
    RSI.rename(columns={symbol:'RSI'}, inplace=True)
    pB = ind.Bollinger_bands(price, lookback=14)
    pB.rename(columns={symbol:'pB'}, inplace=True)
    ppo = ind.prec_price_oscillator(price)
    ppo.rename(columns={symbol:'PPO'}, inplace=True)
   # emv = ind.EMV(price)
   # emv.rename(columns={symbol: 'EMV'}, inplace=True)



 #   bb, bbp, top_band, bottom_band, sma = ind.my_Bollinger_bands(price, lookback=14)

    indicators_df = price.join(RSI).join(pB).join(ppo)#.join(emv)


    #indicators_df['signal'] = indicators_df.apply(lambda x: equation(x), axis=1)
    #indicators_df['signal'] = indicators_df.apply(lambda row: 1 if row['RSI']<=-1.5 and row['pB']<=-1.6  and row['PPO']<-0.5 \
    #           else -1 if row['RSI']>=1.2 and row['pB']>=1.6 and row['PPO']>0 and row['PPO']<1 else 0, axis=1)

    indicators_df['signal'] = signals(indicators_df)
        #pd.concat([price, RSI, pB, ppo])

  #  indicators_df['signal'].plot(label='Signal', color='black', ax=ax1)

    trades = execute_trades(indicators_df, symbol, sd, ed, sv)
    ##################################### PLOT #################################################

    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(4, 1, height_ratios=[1, 0.5,0.5,0.5])
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])
    ax2 = plt.subplot(gs[2])
    ax3 = plt.subplot(gs[3])
    #ax4 = plt.subplot(gs[4])

    price[symbol].plot(label='Price', color='orange', ax=ax0)
    RSI['RSI'].plot(label='RSI', color='blue', ax=ax1)
    pB['pB'].plot(label='%B', color='green', ax=ax2)
    ppo['PPO'].plot(label='PPO', color='red', ax=ax3)
    #emv['EMV'].plot(label='EMV', color='magenta', ax=ax4)

    #top_band[symbol].plot(label='upper-band', color='blue', linestyle='dashed', ax=ax0)
    #bottom_band[symbol].plot(label='lower-band', color='blue', linestyle='dashed', ax=ax0)

    ax0.xaxis.set_major_formatter(DateFormatter('%Y-%m'))
    ax0.xaxis.set_major_locator(mdates.MonthLocator(interval=3))

    ax0.get_xaxis().set_visible(False)
    ax1.get_xaxis().set_visible(False)
    ax2.get_xaxis().set_visible(False)
    #ax3.get_xaxis().set_visible(False)

    ax0.set_ylabel('Stock Price', fontsize=15)
    ax1.set_ylabel('RSI', fontsize=15)
    ax2.set_ylabel('%B', fontsize=15)
    ax3.set_ylabel('PPO', fontsize=15)
  #  ax4.set_ylabel('EMV', fontsize=15)

    ax1.axhline(y=2, linestyle='dashed')
    ax1.axhline(y=-2, linestyle='dashed')
    ax2.axhline(y=2, linestyle='dashed')
    ax2.axhline(y=-2, linestyle='dashed')
    ax3.axhline(y=0, linestyle='dashed')
   # ax4.axhline(y=0, linestyle='dashed')

    ax0.legend()
    ax0.grid()
    plt.xlabel('Date', fontsize=15)
    plt.xticks(rotation=30)
    ax0.set_title(f'Technical indicators', fontsize=22)

    for i, v in trades[symbol].iteritems():
        if v > 0:
            ax0.axvline(x=i, color='green', linestyle='dashed')
            ax1.axvline(x=i, color='green', linestyle='dashed')
            ax2.axvline(x=i, color='green', linestyle='dashed')
            ax3.axvline(x=i, color='green', linestyle='dashed')
        #    ax4.axvline(x=i, color='green', linestyle='dashed')
        elif v < 0:
            ax0.axvline(x=i, color='red', linestyle='dashed')
            ax1.axvline(x=i, color='red', linestyle='dashed')
            ax2.axvline(x=i, color='red', linestyle='dashed')
            ax3.axvline(x=i, color='red', linestyle='dashed')
       #     ax4.axvline(x=i, color='red', linestyle='dashed')

    plt.savefig(f'Indicators and price chart for {symbol} from {sd} to {ed}.png')
    #plt.show()
    ########################################################################################

    return trades



if __name__ == "__main__":
    print("One does not simply think up a strategy")
    #testPolicy(symbol="ML4T-220", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000)
