
  		  	   		   	 		  		  		    	 		 		   		 		  
import datetime as dt  		  	   		   	 		  		  		    	 		 		   		 		  

from marketsimcode import compute_portvals,bechmark
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates
import numpy as np

import StrategyLearner as sl
import ManualStrategy as ms


def author():
  return 'mghoneim3'


def plot_MSvsTSvsBenchMark(manual_trades, ml_trades,bm, symbol, sv):
    '''
    reads manual startegy trades, strategy learner trades
    plots chart showing cumm returns comparing the 2 strategies with benchmark
    '''

    # run marketsim to compute portfolio values
    portvals_manual = compute_portvals(manual_trades, start_val=sv, commission=9.95, impact=0.005)
    portvals_ml = compute_portvals(ml_trades, start_val=sv, commission=9.95, impact=0.005)
    portvals_bm = compute_portvals(bm, start_val=sv, commission=9.95, impact=0.005)


   # portvals_optim.to_csv('portvals_optim.csv')
    #cum_ret_manual = (portvals_manual[-1] / portvals_manual[0] - 1)
    #cum_ret_ml = (portvals_ml[-1] / portvals_ml[0] - 1)
    #cum_ret_bm = (portvals_bm[-1] / portvals_bm[0] - 1)

    fig, ax = plt.subplots(figsize=(10,7))
    plt.plot(portvals_manual / portvals_manual.iloc[0], label='ManualStrategy', color='blue')
    plt.plot(portvals_ml / portvals_ml.iloc[0], label='StrategyLearner', color='red')
    plt.plot(portvals_bm / portvals_bm.iloc[0], label='benchmark', color='green')

    ax.xaxis.set_major_formatter(DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xlabel('Date')
    plt.ylabel('Normalized returns', fontsize=15)
    plt.title(f'ManualvsStrategyLvsBenchmark for {symbol}', fontsize=15)
    plt.legend()
    plt.grid()
    plt.xticks(rotation=30)
    plt.savefig('experiment1.png')
    #plt.show()

def run_exp():
    #np.random.seed(442)

    start_date = dt.datetime(2008, 1, 1)
    end_date = dt.datetime(2009, 12, 31)
    starting_val = 100000
    ticker = "JPM"

    # compute benchmark
    bm = bechmark(start_date, end_date, ticker=ticker)
    if  isinstance(bm, str):
        return 'symbol or date are wrong'

    # generate trades_df for manual strategy
    manual_trades_df = ms.testPolicy(symbol=ticker, sd=start_date, ed=end_date, sv=starting_val)
    # generate trades_df for strategy learner
    SLearner = sl.StrategyLearner(verbose=False, impact=0.0, commission=0.0)  # constructor
    SLearner.add_evidence(symbol=ticker, sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=starting_val)
    ml_trades_df = SLearner.testPolicy(symbol=ticker, sd=start_date, ed=end_date, sv=starting_val)
    # plot

    #SLearner.roll_fwd_cv(symbol=ticker, sd=start_date, ed=end_date)

    plot_MSvsTSvsBenchMark(manual_trades_df, ml_trades_df, bm, ticker, starting_val)

if __name__ == "__main__":  		  	   		   	 		  		  		    	 		 		   		 		  
    run_exp()
