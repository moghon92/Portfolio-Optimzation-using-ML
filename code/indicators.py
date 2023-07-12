import numpy as np
import pandas as pd
import datetime as dt
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates
from matplotlib import gridspec

from util import get_data, plot_data
import matplotlib.pyplot as plt
  		  	   		   	 		  		  		    	 		 		   		 		  
# this function should return a dataset (X and Y) that will work  		  	   		   	 		  		  		    	 		 		   		 		  
# better for linear regression than decision trees  		  	   		   	 		  		  		    	 		 		   		 		  

def author():  		  	   		   	 		  		  		    	 		 		   		 		  
    """  		  	   		   	 		  		  		    	 		 		   		 		  
    :return: The GT username of the student  		  	   		   	 		  		  		    	 		 		   		 		  
    :rtype: str  		  	   		   	 		  		  		    	 		 		   		 		  
    """  		  	   		   	 		  		  		    	 		 		   		 		  
    return "mghoneim3"  # Change this to your user ID
  		  	   		   	 		  		  		    	 		 		   		 		  

def my_momentum(price,lookback=14):
    return price/price.shift(lookback)


def momentum(price,lookback=14):
    mom = price / price.shift(lookback)
    mom_norm = (mom-mom.mean())/mom.std()
    return mom_norm


def SMA(price, lookback=14):
    sma = price.rolling(window=lookback, min_periods=lookback).mean()
    return sma

def price_SMA(price, lookback=14):
    sma = price.rolling(window=lookback, min_periods=lookback).mean()
    return price/sma

def my_Bollinger_bands(price, lookback=14):
    sma = price.rolling(window=lookback, min_periods=lookback).mean()
    std = price.rolling(window=lookback, min_periods=lookback).std()

    top_band = sma + 2*std
    bottom_band = sma - 2*std

    bbp = (price - bottom_band) / (top_band - bottom_band)
    bb = (price - sma)/(2*std)

    #sma_norm = (sma-sma.mean())/sma.std()
    #bbp_norm = (bbp - bbp.mean())/bbp.std()
    #bb_norm = (bb-bb.mean())/bb.std()
    #top_band_norm = (top_band-top_band.mean())/top_band.std()
    #bottom_band_norm = (bottom_band-bottom_band.mean())/bottom_band.std()

    return bb, bbp, top_band,bottom_band, sma

def Bollinger_bands(price, lookback=14):
    sma = price.rolling(window=lookback, min_periods=lookback).mean()
    std = price.rolling(window=lookback, min_periods=lookback).std()

    top_band = sma + 2*std
    bottom_band = sma - 2*std

    bbp = (price - bottom_band) / (top_band - bottom_band)
    bb = (price - sma)/(2*std)

    #sma_norm = (sma-sma.mean())/sma.std()
    bbp_norm = (bbp - bbp.mean())/bbp.std()
    #bb_norm = (bb-bb.mean())/bb.std()
    #top_band_norm = (top_band-top_band.mean())/top_band.std()
    #bottom_band_norm = (bottom_band-bottom_band.mean())/bottom_band.std()
   # print((1 - bbp.mean())/bbp.std())
   # print((0 - bbp.mean()) / bbp.std())

    return bbp_norm

def my_Rel_SI(price, lookback=14):
    daily_rets = price - price.shift(1)
    rsi = pd.DataFrame(index=price.index, data=0, columns=price.columns)

    for day in range(price.shape[0]):
        up_gain = daily_rets.iloc[day-lookback+1:day+1,:].where(daily_rets >= 0).sum()
        down_loss = -1*daily_rets.iloc[day - lookback + 1:day+1, :].where(daily_rets < 0).sum()

        rs=(up_gain/lookback)/(down_loss/lookback)
        rsi.iloc[day,:] = 100-(100/(1+rs))

    rsi[rsi == np.inf] = 100
    rsi_norm = (rsi-rsi.mean())/rsi.std()
    return rsi

def Rel_SI(price, lookback=14):
    daily_rets = price - price.shift(1)
    rsi = pd.DataFrame(index=price.index, data=0, columns=price.columns)

    for day in range(price.shape[0]):
        up_gain = daily_rets.iloc[day-lookback+1:day+1,:].where(daily_rets >= 0).sum()
        down_loss = -1*daily_rets.iloc[day - lookback + 1:day+1, :].where(daily_rets < 0).sum()

        rs=(up_gain/lookback)/(down_loss/lookback)
        rsi.iloc[day,:] = 100-(100/(1+rs))

    rsi[rsi == np.inf] = 100
    rsi_norm = (rsi-rsi.mean())/rsi.std()


    return rsi_norm

def my_prec_price_oscillator(price):
    '''
    Percentage price oscilator
    https://school.stockcharts.com/doku.php?id=technical_indicators:price_oscillators_ppo
    '''
    day12_ema = price.ewm(span=12, min_periods=0, adjust=False, ignore_na=False).mean()
    day26_ema = price.ewm(span=26, min_periods=0, adjust=False, ignore_na=False).mean()
    #day9_ema = df.ewm(span=9, min_periods=0, adjust=False, ignore_na=False).mean()
    ppo = (day12_ema - day26_ema) * 100 / day26_ema
    ppo_signal = ppo.ewm(span=9, min_periods=0, adjust=False, ignore_na=False).mean()
    ppo_hist = ppo-ppo_signal

    pp_norm = (ppo-ppo.mean())/ppo.std()
    return pp_norm, ppo_signal,ppo_hist, day12_ema, day26_ema

def prec_price_oscillator(price):
    '''
    Percentage price oscilator
    https://school.stockcharts.com/doku.php?id=technical_indicators:price_oscillators_ppo
    '''
    day12_ema = price.ewm(span=12, min_periods=0, adjust=False, ignore_na=False).mean()
    day26_ema = price.ewm(span=26, min_periods=0, adjust=False, ignore_na=False).mean()
    #day9_ema = df.ewm(span=9, min_periods=0, adjust=False, ignore_na=False).mean()
    ppo = (day12_ema - day26_ema) * 100 / day26_ema
    ppo_signal = ppo.ewm(span=9, min_periods=0, adjust=False, ignore_na=False).mean()
    ppo_hist = ppo-ppo_signal

    pp_norm = (ppo-ppo.mean())/ppo.std()
    ppo_hist_norm = (ppo_hist-ppo_hist.mean())/ppo_hist.std()
    return ppo_hist_norm

def my_market_correlation(price, lookback=20):
    new_df = price.copy()
    if 'SPY' not in new_df.columns:
        # add SPY
        start_date = new_df.index.min()
        end_date = new_df.index.max()
        market = get_data(['SPY'], pd.date_range(start_date, end_date))

        new_df = new_df.join(market, how='inner')

    corr_df = pd.DataFrame()
    temp = new_df.rolling(window=lookback, min_periods=lookback).corr()['SPY'].reset_index()
    corr_df[new_df.columns] = temp.pivot(index='level_0', columns='level_1', values='SPY')
    corr_df = corr_df[price.columns] # takeout SPY if not in orginal df

    corr_df_norm = (corr_df-corr_df.mean())/corr_df.std()

    return corr_df


def market_correlation(price, lookback=20):
    new_df = price.copy()
    if 'SPY' not in new_df.columns:
        # add SPY
        start_date = new_df.index.min()
        end_date = new_df.index.max()
        market = get_data(['SPY'], pd.date_range(start_date, end_date))

        new_df = new_df.join(market, how='inner')

    corr_df = pd.DataFrame()
    temp = new_df.rolling(window=lookback, min_periods=lookback).corr()['SPY'].reset_index()
    corr_df[new_df.columns] = temp.pivot(index='level_0', columns='level_1', values='SPY')
    corr_df = corr_df[price.columns]  # takeout SPY if not in orginal df

    corr_df_norm = (corr_df - corr_df.mean()) / corr_df.std()

    return corr_df_norm

def EMV(price):
    tickers = price.columns

    start_date = price.index.min()
    end_date = price.index.max()
    df_highs = get_data(tickers, pd.date_range(start_date, end_date), colname="High")
    df_lows = get_data(tickers, pd.date_range(start_date, end_date), colname="Low")
    df_vol = get_data(tickers, pd.date_range(start_date, end_date), colname="Volume")
    df_highs = df_highs[tickers]
    df_lows = df_lows[tickers]
    df_vol = df_vol[tickers]

    distance = ((df_highs + df_lows) / 2 - (df_highs.shift(1) + df_lows.shift(1)) / 2)
    BoxRatio = ((df_vol / 100000000) / (df_highs - df_lows))

    emv = distance / BoxRatio
    emv_smooth = emv.rolling(window=14, min_periods=14).mean()
    emv_norm = (emv_smooth-emv_smooth.mean())/emv_smooth.std()
    #emv_norm.to_frame(name=tickers[0]).fillna(0, inplace=True)
    emv_norm.fillna(0, inplace=True)

    return emv_norm

def my_EMV(price):

    tickers = price.columns

    start_date = price.index.min()
    end_date = price.index.max()
    df_highs = get_data(tickers, pd.date_range(start_date, end_date), colname="High")
    df_lows = get_data(tickers, pd.date_range(start_date, end_date), colname="Low")
    df_vol = get_data(tickers, pd.date_range(start_date, end_date), colname="Volume")
    df_highs = df_highs[tickers]
    df_lows = df_lows[tickers]
    df_vol = df_vol[tickers]

    distance = ((df_highs + df_lows) / 2 - (df_highs.shift(1) + df_lows.shift(1)) / 2)
    BoxRatio = ((df_vol / 100000000) / (df_highs - df_lows))

    emv = distance / BoxRatio
    emv_smooth = emv.rolling(window=14, min_periods=14).mean()

    return emv, emv_smooth, df_vol


### PLOTTing functions ###

def plot_bolligerBands(price, symbol='JPM'):
    bb, bbp, top_band, bottom_band, sma = my_Bollinger_bands(price, lookback=20)

    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 0.5])
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])

    price[symbol].plot(label='Price', color='red', ax=ax0)
    top_band[symbol].plot(label='upper-band', color='blue', linestyle='dashed', ax=ax0)
    bottom_band[symbol].plot(label='lower-band', color='blue', linestyle='dashed', ax=ax0)
    sma[symbol].plot(label='20-day SMA', color='orange', ax=ax0)

    bbp[symbol].plot(label='bbp', color='green', ax=ax1)

    ax0.xaxis.set_major_formatter(DateFormatter('%Y-%m'))
    ax0.xaxis.set_major_locator(mdates.MonthLocator(interval=3))

    ax0.set_ylabel('Stock Price', fontsize=15)
    ax1.set_ylabel('%B indicator', fontsize=15)
    ax1.axhline(y=0, linestyle='dashed')
    ax1.axhline(y=1, linestyle='dashed')
    ax0.legend()
    ax0.grid()
    plt.xlabel('Date', fontsize=15)
    plt.xticks(rotation=30)
    ax0.set_title(f'Bollinger Bands & Bbp', fontsize=22)
    plt.savefig('bollinger_bands.png')


def plot_SPY_correlation(price,price_SPY, symbol='JPM'):
    corr = my_market_correlation(price, lookback=20)

    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 0.5])
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])

    price_norm = (price/price.iloc[0]) - 1
    price_spy_nomr = (price_SPY/price_SPY.iloc[0]) - 1

    price_norm[symbol].plot(label=symbol, color='blue', ax=ax0)
    price_spy_nomr['SPY'].plot(label='SPY', color='red', ax=ax0)
    #day26_ema[symbol].plot(label='26dayEMA', color='black')

    corr[symbol].plot(color='green', ax=ax1)
    #ppo_signal[symbol].plot(label='9dayEMAPPO', color='red', ax=ax1)
    #ppo_hist[symbol].plot.hist(color='blue', ax=ax1)


    ax0.xaxis.set_major_formatter(DateFormatter('%Y-%m'))
    ax0.xaxis.set_major_locator(mdates.MonthLocator(interval=3))

    ax0.set_ylabel('Stock Price', fontsize=15)
    ax1.set_ylabel('Corr. with SPY', fontsize=15)
    ax1.axhline(y=0, linestyle='dashed')
    ax0.legend()
    ax0.grid()
    plt.xlabel('Date', fontsize=15)
    plt.xticks(rotation=30)
    ax0.set_title(f'Correlation with SPY', fontsize=22)
    plt.savefig('correlation_with_SPY.png')


def plot_PPO(price, symbol='JPM'):
    ppo, ppo_signal,ppo_hist,day12_ema,day26_ema= my_prec_price_oscillator(price)

    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 0.5])
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])

    price[symbol].plot(label='stock Price', color='green', ax=ax0)
    day12_ema[symbol].plot(label='12dayEMA', color='red', ax=ax0)
    day26_ema[symbol].plot(label='26dayEMA', color='black', ax=ax0)

    ppo[symbol].plot(color='magenta', ax=ax1)
    #ppo_signal[symbol].plot(label='9dayEMAPPO', color='red', ax=ax1)
    #ppo_hist[symbol].plot.hist(color='blue', ax=ax1)


    ax0.xaxis.set_major_formatter(DateFormatter('%Y-%m'))
    ax0.xaxis.set_major_locator(mdates.MonthLocator(interval=3))

    ax0.set_ylabel('Stock Price', fontsize=15)
    ax1.set_ylabel('PPO', fontsize=15)
    ax1.axhline(y=0, linestyle='dashed')
    ax0.legend()
    ax0.grid()
    plt.xlabel('Date', fontsize=15)
    plt.xticks(rotation=30)
    ax0.set_title(f'Percentage Price Oscilator', fontsize=22)
    plt.savefig('PPO.png')



def plot_Momentum(price, symbol='JPM'):
    mom= my_momentum(price)

    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 0.5])
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])

    price[symbol].plot(label='stock Price', color='green', ax=ax0)
    mom[symbol].plot(color='magenta', ax=ax1)
    #ppo_signal[symbol].plot(label='9dayEMAPPO', color='red', ax=ax1)
    #ppo_hist[symbol].plot.hist(color='blue', ax=ax1)


    ax0.xaxis.set_major_formatter(DateFormatter('%Y-%m'))
    ax0.xaxis.set_major_locator(mdates.MonthLocator(interval=3))

    ax0.set_ylabel('Stock Price', fontsize=15)
    ax1.set_ylabel('Momentum', fontsize=15)
    #ax1.axhline(y=0, linestyle='dashed')
    ax0.legend()
    ax0.grid()
    plt.xlabel('Date', fontsize=15)
    plt.xticks(rotation=30)
    ax0.set_title(f'Momentum', fontsize=22)
    plt.savefig('Momentum.png')


def plot_EMV(price, symbol='JPM'):
    emv, emv_smooth, vol = my_EMV(price)


    fig = plt.figure(figsize=(20, 15))
    gs = gridspec.GridSpec(3, 1, height_ratios=[1, 0.5, 0.5])
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])
    ax2 = plt.subplot(gs[2])

    price[symbol].plot(label=symbol, color='green', ax=ax0)

    vol.plot.bar(label='volume', color='blue', ax=ax1)
    emv_smooth[symbol].plot(color='red', ax=ax2)

    #ppo_signal[symbol].plot(label='9dayEMAPPO', color='red', ax=ax1)
    #ppo_hist[symbol].plot.hist(color='blue', ax=ax1)


    ax0.xaxis.set_major_formatter(DateFormatter('%Y-%m'))
    ax0.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax1.xaxis.set_major_formatter(DateFormatter('%Y-%m'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))

    ax0.set_ylabel('Stock Price', fontsize=15)
    ax1.set_ylabel('Volume', fontsize=15)
    ax2.set_ylabel('EMV', fontsize=15)
    ax2.axhline(y=0, linestyle='dashed')


    ax0.legend()
    ax0.grid()
    plt.xlabel('Date', fontsize=15)
    plt.xticks(rotation=30)
    ax0.set_title(f'Ease of Movement (EMV)', fontsize=22)
    plt.savefig('EMV.png')




def plot_RSI(price, symbol='JPM'):
    rsi = my_Rel_SI(price, lookback=14)

    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 0.5])
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])

    price[symbol].plot(label='stock Price', color='green', ax=ax0)


    rsi[symbol].plot(color='red', ax=ax1)
    #ppo_signal[symbol].plot(label='9dayEMAPPO', color='red', ax=ax1)
    #ppo_hist[symbol].plot.hist(color='blue', ax=ax1)


    ax0.xaxis.set_major_formatter(DateFormatter('%Y-%m'))
    ax0.xaxis.set_major_locator(mdates.MonthLocator(interval=3))

    ax0.set_ylabel('Stock Price', fontsize=15)
    ax1.set_ylabel('RSI', fontsize=15)
    ax1.axhline(y=30, linestyle='dashed')
    ax1.axhline(y=70, linestyle='dashed')

    ax0.legend()
    ax0.grid()
    plt.xlabel('Date', fontsize=15)
    plt.xticks(rotation=30)
    ax0.set_title(f'Relaive Strength Index', fontsize=22)
    plt.savefig('RSI.png')


def main():
    start_date = dt.datetime(2008, 1, 1)
    end_date = dt.datetime(2009, 12, 31)
    symbol = 'JPM'
    tickers = [symbol]

    price = get_data(tickers, pd.date_range(start_date, end_date))
    price_ticker = price[tickers]
    price_SPY = price[['SPY']]

    plot_bolligerBands(price_ticker, symbol='JPM')
    plot_PPO(price_ticker, symbol='JPM')
    plot_SPY_correlation(price_ticker, price_SPY, symbol='JPM')
    plot_EMV(price_ticker, symbol='JPM')
    plot_RSI(price_ticker, symbol='JPM')


# plot_Momentum(price, symbol='JPM')

if __name__ == "__main__":
    main()

