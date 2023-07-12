
  		  	   		   	 		  		  		    	 		 		   		 		  
import datetime as dt
import time
from marketsimcode import compute_portvals,bechmark
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates
import numpy as np

import ManualStrategy as ms

import experiment1 as exp1
import experiment2 as exp2



def author():
  return 'mghoneim3'


def report_plots(symbol, start_date, end_date, sv):
    '''
    function to generate the plots required for manual strategy for report
    also generates the data comparing in and out of sample performance
    '''

    # use market sim to comptue part values
    bm = bechmark(start_date, end_date, ticker=symbol)
    if isinstance(bm, str):
        return 'symbol or date are wrong'

    manual_trades = ms.testPolicy_with_plot(symbol=symbol, sd=start_date, ed=end_date, sv=sv)
    portvals_manual = compute_portvals(manual_trades, start_val=sv, commission=9.95, impact=0.005)
    portvals_bm = compute_portvals(bm, start_val=sv, commission=9.95, impact=0.005)

    # calculate portfolio metrics
    cum_ret_manual = (portvals_manual[-1] / portvals_manual[0] - 1)
    cum_ret_bm = (portvals_bm[-1] / portvals_bm[0] - 1)

    daily_rets_manual = (portvals_manual / portvals_manual.shift(1)) - 1
    daily_rets_manual = daily_rets_manual[1:]
    avg_daily_ret_manual = daily_rets_manual.mean()
    std_daily_ret_manual = daily_rets_manual.std()

    daily_rets_bm = (portvals_bm / portvals_bm.shift(1)) - 1
    daily_rets_bm = daily_rets_bm[1:]
    avg_daily_ret_bm = daily_rets_bm.mean()
    std_daily_ret_bm = daily_rets_bm.std()

    # write meetrics to txt file
    with open('manualStrategyVsBenchmark.txt', 'a') as f:
        f.writelines('\n')
        f.writelines(f'\n{symbol} from {start_date} to {end_date} :-')
        f.writelines(f'\nCumm Return manual = {cum_ret_manual}, cumm return bm = {cum_ret_bm}')
        f.writelines(f'\nstd daily Return manual = {std_daily_ret_manual}, std daily return bm = {std_daily_ret_bm}')
        f.writelines(f'\nmean daily Return manual = {avg_daily_ret_manual}, mean daily return bm = {avg_daily_ret_bm}')

  #  print()
  #  print(f'Cumm Return manual = {cum_ret_manual}, cumm return bm = {cum_ret_bm}')
  #  print(f'std daily Return manual = {std_daily_ret_manual}, std daily return bm = {std_daily_ret_bm}')
  #  print(f'mean daily Return manual = {avg_daily_ret_manual}, mean daily return bm = {avg_daily_ret_bm}')

    # plot chart to compare manual stargey to benchmark
    fig, ax = plt.subplots(figsize=(10, 7))
    plt.plot(portvals_manual / portvals_manual.iloc[0], label='ManualStrategy', color='red')
    plt.plot(portvals_bm / portvals_bm.iloc[0], label='Benchmark', color='green')

    ax.xaxis.set_major_formatter(DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xlabel('Date')
    plt.ylabel('Normalized returns', fontsize=15)
    plt.title(f'ManualvsBenchmark for {symbol}', fontsize=15)
    plt.legend()
    plt.grid()
    plt.xticks(rotation=30)

    for i, v in manual_trades[symbol].iteritems():
        if v > 0:
            ax.axvline(x=i, color='blue', linestyle='dashed')

        elif v <0 :
            ax.axvline(x=i, color='black', linestyle='dashed')
    plt.savefig(f'ManualStareyvsBenchMark for {symbol} from {start_date} to {end_date}.png')
   # plt.show()


if __name__ == "__main__":  		  	   		   	 		  		  		    	 		 		   		 		  
    np.random.seed(442)

    # plot manual startegy vs becnmark for in-sample
    report_plots('JPM', dt.datetime(2008, 1, 1), dt.datetime(2009, 12, 31), 100000)
    # plot manual startegy vs becnmark for out of sample
    report_plots('JPM', dt.datetime(2010, 1, 1), dt.datetime(2011, 12, 31), 100000)

    t1 = time.time()
    exp1.run_exp()
    t2 = time.time()
   # print('exp1 executed in :', t2 - t1)
    exp2.run_exp()
    t3 = time.time()
   # print('exp2 executed in :', t3 - t2)
   # print('all executed in :', t3 - t1)

