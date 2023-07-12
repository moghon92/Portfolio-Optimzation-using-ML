
  		  	   		   	 		  		  		    	 		 		   		 		  
import datetime as dt
import pandas as pd  		  	   		   	 		  		  		    	 		 		   		 		  

from marketsimcode import compute_portvals,bechmark
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates
import numpy as np

import StrategyLearner as sl

def author():
  return 'mghoneim3'

def run_exp():
    start_date = dt.datetime(2008, 1, 1)
    end_date = dt.datetime(2009, 12, 31)
    starting_val = 100000
    ticker = "JPM"

    fig, ax = plt.subplots(figsize=(10,7))

    # compute benchmark portfolio values
    bm = bechmark(start_date, end_date, ticker=ticker)
    if  isinstance(bm, str):
        return 'symbol or date are wrong'

    portvals_bm = compute_portvals(bm, start_val=starting_val, commission=9.95, impact=0.005)

    # set the 3 impact levels that will be tested
    impacts = [0.0,0.005, 0.05]
    colors = ['red','orange', 'magenta']

    cum_returns = []
    sharp_ratios = []

    # loop through the impact levels and record cum_ret and sharpe_ratio
    for i in range(3):
        SLearner = sl.StrategyLearner(verbose=False, impact=impacts[i], commission=0.0)  # constructor
        SLearner.add_evidence(symbol=ticker, sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31),sv=starting_val)
        ml_trades_df = SLearner.testPolicy(symbol=ticker, sd=start_date, ed=end_date, sv=starting_val)

        portvals_ml = compute_portvals(ml_trades_df, start_val=starting_val, commission=9.95, impact=0.005)
        plt.plot(portvals_ml / portvals_ml.iloc[0], label=f'impact={impacts[i]}', color=colors[i])

        daily_rets_optim = (portvals_ml / portvals_ml.shift(1)) - 1
        daily_rets_optim = daily_rets_optim[1:]
        cum_ret = (portvals_ml[-1] / portvals_ml[0] - 1)
        avg_daily_ret = daily_rets_optim.mean()
        std_daily_ret = daily_rets_optim.std()
        sharpe_ratio = np.sqrt(252) * avg_daily_ret / std_daily_ret

        cum_returns.append(cum_ret)
        sharp_ratios.append(sharpe_ratio)


    # plot a chart showing how normalized return changes for each impact level
    plt.plot(portvals_bm / portvals_bm.iloc[0], label='benchmark', color='green')
    ax.xaxis.set_major_formatter(DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xlabel('Date', fontsize=15)
    plt.ylabel('Normalized returns', fontsize=15)
    plt.title(f'Cumm Returns for diff impacts on {ticker}', fontsize=15)
    plt.legend()
    plt.grid()
    plt.xticks(rotation=30)
    plt.savefig('experiemnt2 benchmark.png')
    #plt.show()

    # plot a bar chart showing the cumm return and sharp ratios as bar chart
    fig2, ax2 = plt.subplots(figsize=(15,10))
    df = pd.DataFrame(index=impacts, data=0.0, columns=['Cumm_Return','Sharp_Ratio'])
    df['Cumm_Return'] = pd.Series(cum_returns, index=impacts)
    df['Sharp_Ratio']= pd.Series(sharp_ratios, index=impacts)

    ax2=df.T.plot.bar(rot=0)


    plt.ylabel('value', fontsize=15)
    plt.title(f'Cumm Returns & sharpRatio for diff impact lvls on {ticker}', fontsize=10)
    plt.grid()
    plt.savefig('experiemnt2 comparison.png')
    #plt.show()

if __name__ == "__main__":  		  	   		   	 		  		  		    	 		 		   		 		  
    run_exp()
