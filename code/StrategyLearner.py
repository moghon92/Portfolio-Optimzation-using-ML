""""""  		  	   		   	 		  		  		    	 		 		   		 		  
"""  		  	   		   	 		  		  		    	 		 		   		 		  
Template for implementing StrategyLearner  (c) 2016 Tucker Balch  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		   	 		  		  		    	 		 		   		 		  
Atlanta, Georgia 30332  		  	   		   	 		  		  		    	 		 		   		 		  
All Rights Reserved  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
Template code for CS 4646/7646  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		   	 		  		  		    	 		 		   		 		  
works, including solutions to the projects assigned in this course. Students  		  	   		   	 		  		  		    	 		 		   		 		  
and other users of this template code are advised not to share it with others  		  	   		   	 		  		  		    	 		 		   		 		  
or to make it available on publicly viewable websites including repositories  		  	   		   	 		  		  		    	 		 		   		 		  
such as github and gitlab.  This copyright statement should not be removed  		  	   		   	 		  		  		    	 		 		   		 		  
or edited.  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
We do grant permission to share solutions privately with non-students such  		  	   		   	 		  		  		    	 		 		   		 		  
as potential employers. However, sharing with other current or future  		  	   		   	 		  		  		    	 		 		   		 		  
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		   	 		  		  		    	 		 		   		 		  
GT honor code violation.  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
-----do not edit anything above this line---  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
Student Name: Tucker Balch (replace with your name)  		  	   		   	 		  		  		    	 		 		   		 		  
GT User ID: tb34 (replace with your User ID)  		  	   		   	 		  		  		    	 		 		   		 		  
GT ID: 900897987 (replace with your GT ID)  		  	   		   	 		  		  		    	 		 		   		 		  
"""  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
import datetime as dt  		  	   		   	 		  		  		    	 		 		   		 		  
import math

  		  	   		   	 		  		  		    	 		 		   		 		  
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import util as ut

import indicators as ind


import RTLearner as rt
import BagLearner as bl

class StrategyLearner(object):  		  	   		   	 		  		  		    	 		 		   		 		  
    """  		  	   		   	 		  		  		    	 		 		   		 		  
    A strategy learner that can learn a trading policy using the same indicators used in ManualStrategy.  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
    :param verbose: If “verbose” is True, your code can print out information for debugging.  		  	   		   	 		  		  		    	 		 		   		 		  
        If verbose = False your code should not generate ANY output.  		  	   		   	 		  		  		    	 		 		   		 		  
    :type verbose: bool  		  	   		   	 		  		  		    	 		 		   		 		  
    :param impact: The market impact of each transaction, defaults to 0.0  		  	   		   	 		  		  		    	 		 		   		 		  
    :type impact: float  		  	   		   	 		  		  		    	 		 		   		 		  
    :param commission: The commission amount charged, defaults to 0.0  		  	   		   	 		  		  		    	 		 		   		 		  
    :type commission: float  		  	   		   	 		  		  		    	 		 		   		 		  
    """  		  	   		   	 		  		  		    	 		 		   		 		  
    # constructor  		  	   		   	 		  		  		    	 		 		   		 		  
    def __init__(self, verbose=False, impact=0.0, commission=0.0):  		  	   		   	 		  		  		    	 		 		   		 		  
        """  		  	   		   	 		  		  		    	 		 		   		 		  
        Constructor method  		  	   		   	 		  		  		    	 		 		   		 		  
        """  		  	   		   	 		  		  		    	 		 		   		 		  
        self.verbose = verbose  		  	   		   	 		  		  		    	 		 		   		 		  
        self.impact = impact  		  	   		   	 		  		  		    	 		 		   		 		  
        self.commission = commission
        self.learner = []
  		  	   		   	 		  		  		    	 		 		   		 		  
    # this method should create a QLearner, and train it for trading  		  	   		   	 		  		  		    	 		 		   		 		  
    def add_evidence(  		  	   		   	 		  		  		    	 		 		   		 		  
        self,  		  	   		   	 		  		  		    	 		 		   		 		  
        symbol="IBM",  		  	   		   	 		  		  		    	 		 		   		 		  
        sd=dt.datetime(2008, 1, 1),  		  	   		   	 		  		  		    	 		 		   		 		  
        ed=dt.datetime(2009, 1, 1),  		  	   		   	 		  		  		    	 		 		   		 		  
        sv=10000,  		  	   		   	 		  		  		    	 		 		   		 		  
    ):  		  	   		   	 		  		  		    	 		 		   		 		  
        """  		  	   		   	 		  		  		    	 		 		   		 		  
        Trains your strategy learner over a given time frame.  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
        :param symbol: The stock symbol to train on  		  	   		   	 		  		  		    	 		 		   		 		  
        :type symbol: str  		  	   		   	 		  		  		    	 		 		   		 		  
        :param sd: A datetime object that represents the start date, defaults to 1/1/2008  		  	   		   	 		  		  		    	 		 		   		 		  
        :type sd: datetime  		  	   		   	 		  		  		    	 		 		   		 		  
        :param ed: A datetime object that represents the end date, defaults to 1/1/2009  		  	   		   	 		  		  		    	 		 		   		 		  
        :type ed: datetime  		  	   		   	 		  		  		    	 		 		   		 		  
        :param sv: The starting value of the portfolio  		  	   		   	 		  		  		    	 		 		   		 		  
        :type sv: int  		  	   		   	 		  		  		    	 		 		   		 		  
        """  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
       # t1 = time.time()
        # example usage of the old backward compatible util function  		  	   		   	 		  		  		    	 		 		   		 		  
        syms = [symbol]  		  	   		   	 		  		  		    	 		 		   		 		  
        dates = pd.date_range(sd, ed)

        try:
            prices_all = ut.get_data(syms, dates)  # automatically adds SPY
        except:
            return 'sorry symbol or date ranges does not exist'

        prices = prices_all[syms]  # only portfolio symbols
        prices_SPY = prices_all["SPY"]  # only SPY, for comparison later  		  	   		   	 		  		  		    	 		 		   		 		  
        if self.verbose:  		  	   		   	 		  		  		    	 		 		   		 		  
            print(prices)  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
        # calculate the technical indicators for computing trainingset
        RSI = ind.Rel_SI(prices, lookback=14)
        RSI.rename(columns={symbol: 'RSI'}, inplace=True)
        pB = ind.Bollinger_bands(prices, lookback=14)
        pB.rename(columns={symbol: 'pB'}, inplace=True)
        ppo = ind.prec_price_oscillator(prices)
        ppo.rename(columns={symbol: 'PPO'}, inplace=True)
        #emv = ind.EMV(prices)
        #emv.rename(columns={symbol: 'EMV'}, inplace=True)
        indicators_df = prices.join(RSI).join(pB).join(ppo)#.join(emv)

        # create the y-label for the data by looking N=14 days fwd
        indicators_df['N_day_price'] = indicators_df[symbol].shift(-14)
        indicators_df['N_day_change'] = indicators_df.apply(lambda x: 1 if x['N_day_price']>x[symbol] else -1 if x['N_day_price']<x[symbol] else 0, axis=1)
        indicators_df['commission'] = self.commission
        indicators_df['impact'] = self.impact*indicators_df[symbol]*indicators_df['N_day_change']
        indicators_df['daily_rets'] = ((indicators_df['N_day_price']-indicators_df['commission']-indicators_df['impact'])/indicators_df[symbol])-1.0
        indicators_df['labels'] = indicators_df['daily_rets'].apply(lambda x: 1 if x>0.025 else -1 if x<-0.025 else 0)

        data = indicators_df.dropna()
        del data[symbol]
        del data['N_day_price']
        del data['N_day_change']
        del data['commission']
        del data['impact']
        del data['daily_rets']


        # separate data to train_x and train_y
        train_x = data.iloc[:, 0:-1].values
        train_y = data.iloc[:, -1].values

        # call the baglearner with RT learner
        self.learner = bl.BagLearner(learner=rt.RTLearner, kwargs={"leaf_size": 5}, bags=20, boost=False, verbose=False)
        self.learner.add_evidence(train_x, train_y)

        # predict
        pred_y = self.learner.query(train_x)  # get the predictions
        #rmse_train = math.sqrt(((train_y - pred_y) ** 2).sum() / train_y.shape[0])

     #   t2 = time.time()
     #   print()
     #   print('add_evidence takes: ' , t2-t1)
        #print(rmse_train)


    # this method should use the existing policy and test it against new data
    def testPolicy(  		  	   		   	 		  		  		    	 		 		   		 		  
        self,  		  	   		   	 		  		  		    	 		 		   		 		  
        symbol="IBM",  		  	   		   	 		  		  		    	 		 		   		 		  
        sd=dt.datetime(2009, 1, 1),  		  	   		   	 		  		  		    	 		 		   		 		  
        ed=dt.datetime(2010, 1, 1),  		  	   		   	 		  		  		    	 		 		   		 		  
        sv=10000,  		  	   		   	 		  		  		    	 		 		   		 		  
    ):  		  	   		   	 		  		  		    	 		 		   		 		  
        """  		  	   		   	 		  		  		    	 		 		   		 		  
        Tests your learner using data outside of the training data  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
        :param symbol: The stock symbol that you trained on on  		  	   		   	 		  		  		    	 		 		   		 		  
        :type symbol: str  		  	   		   	 		  		  		    	 		 		   		 		  
        :param sd: A datetime object that represents the start date, defaults to 1/1/2008  		  	   		   	 		  		  		    	 		 		   		 		  
        :type sd: datetime  		  	   		   	 		  		  		    	 		 		   		 		  
        :param ed: A datetime object that represents the end date, defaults to 1/1/2009  		  	   		   	 		  		  		    	 		 		   		 		  
        :type ed: datetime  		  	   		   	 		  		  		    	 		 		   		 		  
        :param sv: The starting value of the portfolio  		  	   		   	 		  		  		    	 		 		   		 		  
        :type sv: int  		  	   		   	 		  		  		    	 		 		   		 		  
        :return: A DataFrame with values representing trades for each day. Legal values are +1000.0 indicating  		  	   		   	 		  		  		    	 		 		   		 		  
            a BUY of 1000 shares, -1000.0 indicating a SELL of 1000 shares, and 0.0 indicating NOTHING.  		  	   		   	 		  		  		    	 		 		   		 		  
            Values of +2000 and -2000 for trades are also legal when switching from long to short or short to  		  	   		   	 		  		  		    	 		 		   		 		  
            long so long as net holdings are constrained to -1000, 0, and 1000.  		  	   		   	 		  		  		    	 		 		   		 		  
        :rtype: pandas.DataFrame  		  	   		   	 		  		  		    	 		 		   		 		  
        """
      #  t1 = time.time()
        dates = pd.date_range(sd, ed)
        try:
            prices_all = ut.get_data([symbol],dates)  # automatically adds SPY
        except:
            return 'sorry symbol or date ranges does not exist'

        trades = prices_all[[symbol, ]]  # only portfolio symbols
        #trades_SPY = prices_all["SPY"]  # only SPY, for comparison later

        # calculate the technical indicators for computing testset
        RSI = ind.Rel_SI(trades, lookback=14)
        RSI.rename(columns={symbol: 'RSI'}, inplace=True)
        pB = ind.Bollinger_bands(trades, lookback=14)
        pB.rename(columns={symbol: 'pB'}, inplace=True)
        ppo = ind.prec_price_oscillator(trades)
        ppo.rename(columns={symbol: 'PPO'}, inplace=True)
       # emv = ind.EMV(trades)
       # emv.rename(columns={symbol: 'EMV'}, inplace=True)

        indicators_df = trades.join(RSI).join(pB).join(ppo)#.join(emv)

       # indicators_df['daily_rets'] = (indicators_df[symbol].shift(-14) / indicators_df[symbol]) - 1.0
       # indicators_df['labels'] = indicators_df['daily_rets'].apply(lambda x: 1 if x > 0.025 else -1 if x < -0.025 else 0)


        data = indicators_df.dropna()
        del data[symbol]
       # del data['daily_rets']

        # put data into numpy array
        test_x = data.iloc[:, :].values
       # test_y = data.iloc[:, -1].values

        # predict
        pred_y = self.learner.query(test_x)  # get the predictions
        #rmse_test = math.sqrt(((test_y - pred_y) ** 2).sum() / test_y.shape[0])
        #print(rmse_test)


        df = data.copy()

        df['preds']  = pred_y
        df['trade_amount'] = 0

        # execute the trades while maintaining position
        net_holdings = 0
        for i in range(df.shape[0]):
            if df.iloc[i, 3] == 1:  # long
                for trade_val in [2000, 1000, 0]:
                    if net_holdings + trade_val in [-1000, 0, 1000]:
                        df.iloc[i, 4] = trade_val
                        net_holdings += trade_val
                        break
            elif df.iloc[i, 3] == -1:
                for trade_val in [-2000, -1000, 0]:
                    if net_holdings + trade_val in [-1000, 0, 1000]:
                        df.iloc[i, 4] = trade_val
                        net_holdings += trade_val
                        break

        out = df.iloc[:, 4]
        ##############################################################
     #   fig = plt.figure(figsize=(15, 10))
     #   indicators_df[symbol].plot(label='Price', color='black')

      #  for i, v in df.iloc[:, 5].iteritems():
      #      if v > 0:
      #          plt.axvline(x=i, color='green', linestyle='dashed')
      #      elif v < 0:
      #          plt.axvline(x=i, color='red', linestyle='dashed')
      #  plt.show()
        ##############################################################

       # t2= time.time()
       # print('test_policy takes: ',  t2-t1)
        return out[:ed].to_frame(name=symbol)


    def roll_fwd_cv(self,symbol="IBM", sd=dt.datetime(2009, 1, 1), ed=dt.datetime(2010, 1, 1)):
        '''
        function developed to detrmine the optimal leafsize using roll fwd cv
        '''
        dates = pd.date_range(sd, ed)
        prices_all = ut.get_data([symbol], dates)  # automatically adds SPY
        trades = prices_all[[symbol, ]]
        RSI = ind.Rel_SI(trades, lookback=14)
        RSI.rename(columns={symbol: 'RSI'}, inplace=True)
        pB = ind.Bollinger_bands(trades, lookback=14)
        pB.rename(columns={symbol: 'pB'}, inplace=True)
        ppo = ind.prec_price_oscillator(trades)
        ppo.rename(columns={symbol: 'PPO'}, inplace=True)
        # emv = ind.EMV(trades)
        # emv.rename(columns={symbol: 'EMV'}, inplace=True)

        indicators_df = trades.join(RSI).join(pB).join(ppo)  # .join(emv)

        # indicators_df['daily_rets'] = (indicators_df[symbol].shift(-14) / indicators_df[symbol]) - 1.0
        # indicators_df['labels'] = indicators_df['daily_rets'].apply(lambda x: 1 if x > 0.025 else -1 if x < -0.025 else 0)
        indicators_df['N_day_price'] = indicators_df[symbol].shift(-14)
        indicators_df['N_day_change'] = indicators_df.apply(lambda x: 1 if x['N_day_price']>x[symbol] else -1 if x['N_day_price']<x[symbol] else 0, axis=1)
        indicators_df['commission'] = self.commission
        indicators_df['impact'] = self.impact*indicators_df[symbol]*indicators_df['N_day_change']
        indicators_df['daily_rets'] = ((indicators_df['N_day_price']-indicators_df['commission']-indicators_df['impact'])/indicators_df[symbol])-1.0
        indicators_df['labels'] = indicators_df['daily_rets'].apply(lambda x: 1 if x>0.025 else -1 if x<-0.025 else 0)

        data = indicators_df.dropna()
        del data[symbol]
        del data['N_day_price']
        del data['N_day_change']
        del data['commission']
        del data['impact']
        del data['daily_rets']

        data['cv'] = pd.qcut(data.index, [0., .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.],
                                         labels=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        # del data['daily_rets'
        bag_train_RMSE_list = []
        for i in range(3,15):
            learner = bl.BagLearner(learner=rt.RTLearner, kwargs={"leaf_size": i}, bags=20, boost=False, verbose=False)
            rmses = []
            for j in range(1,9):
                train_x = data[data['cv']==j].iloc[:, :-2].values
                test_x = data[data['cv'] == j+1].iloc[:, :-2].values
                train_y = data[data['cv']==j]['labels'].values
                test_y = data[data['cv'] == j+1]['labels'].values
                learner.add_evidence(train_x, train_y)
                pred_y = learner.query(test_x)  # get the predictions
                rmse_train = math.sqrt(((test_y - pred_y) ** 2).sum() / train_y.shape[0])
                rmses.append(rmse_train)

            bag_train_RMSE_list.append(np.mean(rmses))

        f2 = plt.figure(figsize=(10,10))
        plt.plot(bag_train_RMSE_list)
        plt.xlabel('leaf size')
        plt.ylabel('RMSE')
        plt.title('RMS per leaf size using roll fwd CV')
        plt.legend()
        plt.savefig('cross-validation.png')
        #plt.show()

        return 1.0


    def author(self):
        return 'mghoneim3'

if __name__ == "__main__":  		  	   		   	 		  		  		    	 		 		   		 		  
    print("One does not simply think up a strategy")
