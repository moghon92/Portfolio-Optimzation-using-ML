
Project 8: STRATEGY EVALUATION

Mohamed Ghoneim
<mghoneim3@gatech.edu>



1. # **INTRODUCION**
In this report we compile together multiple learnings from the course, whereby we use technical indicators to develop a manual strategy that combines signals from 3 different indicators together to generate automated Longs, Short, Hold signals.

We also implement a machine learning based strategy that uses a random forest ML model to make trading decisions. The ML model is trained on the same set of technical indicators that are used for manual strategy so that we can do an apple-to-apple comparison.

The performance of manual strategy of JPM stock is tested over the in-sample period 01-01-2008 to 31-12-2009 against the random forest-based strategy against a benchmark buy and hold strategy. The hypothesis is that strategy learner will outperform manual learner and benchmark over in sample period due to its ability to tease out finer levels of details and capitalize on more trading opportunities. I also expect that manual strategy will outperform the benchmark over in-sample period and (hopefully) if it’s a good strategy it will also outperform on out of sample period (please see section 3 for results).

Lastly, we will conduct an analysis to see if effect of training the strategy learner on different impact levels and how the performance is going to be affected. My expectation is that the higher the impact the more conservative the model will become and will capitalize on less trading opportunities and therefore will have lower returns.
1. # **INDICATOR OVERVIEW**
   1. ## **Relative Strength Index (RSI)**
RSI is a momentum oscillator that can be used to evaluate overbought or oversold conditions in the price of a stock. RSI oscillates between zero and 100. Stock is considered overbought when RSI above 70 which may point towards an upcoming trend reversal which can indicate a sell signal. Conversly, stock is considered oversold when below 30 which can indicate a Buy signal.

In my manual strategy and staregy learner I used a lookback period of 14 days to calculate the RSI signal line. For my manual startegy I choose my oversold signal line to be at 22.5 and overbought signal line to be at 76.5 (which is -2 and +2 on the normalized scale).
1. ## **Bollinger Band Percentage (%B)**
%B quantifies a stock's price relative to the upper and lower Bollinger Bands®. If %B is below 0 means the stock price dropped below the lower band. Conversely when the %B indicator is greater than 1 this means that the price went above the upper band. 

For my manual strategy and strategy learner I used a 14-day lookback period to calculate the %B signal (using rolling mean and standard deviation). For my manual strategy I optimized the use of the %B signal so that buy signals are only generated when %B indicator crosses from -ve to +ve at least twice within 60-day period. This for me is a confirmation that the price is reaching a major support level and is about to reverse. Also, sell signals are generated only when %B signals goes from above 1 to below 1 at least twice within 60 days period. <a name="_hlk88503258"></a>This for me is a confirmation that the price is reaching a major resistance level and is about to reverse. I do not rely on these signals alone for my trading decisions, I use it in combination with other signals as discussed in next chapter.
1. ## **Percentage Price Oscillator (PPO)**
The Percentage Price Oscillator (PPO) is a momentum oscillator that measures the difference between two exponential moving averages (EMA) as a percentage of the larger moving average. PPO indicator moves into positive territory as the shorter moving average distances itself from the longer moving average, this reflects an upside momentum. The PPO is negative when the shorter moving average is below the longer moving average which reflects downside momentum. Buy and sell signals can be generated when the PPO signal or PPO histogram cross the zero line.

Form my manual strategy and strategy learner I used the difference between a 12-day EMA and 26-day EMA to calculate PPO signal. The final output signal was the PPO histogram signal, which is the difference between PPO signal and it’s 9-day EMA.
1. # **MANUAL STRAGETGY**
My Manual trading strategy involves the combination of the 3 technical indicators mentioned in the previous chapter in a relatively complex way to make (long-short-hold) trading decisions. I selected the 3 indicators mentioned above because they can serve multiple purposes at the same time. For example, RSI, %B and PPO can be used as Overbought/oversold indicators and Momentum/trend capturing indicators at the same time by observing their signal line trends. This gave me some flexibility to develop a relatively complex trading strategy. My Manual Trading strategy is a rules-based strategy which generates long/short signals only when certain conditions are met. The strategy goes as follows:

- **Long** when below conditions are met:

|<p>1- PPO histogram signal crossed from negative territory into positive territory</p><p>**AND**</p><p>2- RSI signal went from below 22.5 to above 22.5 (oversold) at least once in the past 28 days **OR** %B signal went from -ve to +ve (price broke back into the lower band) at least twice in the past 60-days.</p>|
| - |
- **Short** when below conditions are met:

|<p>1- PPO histogram signal crossed from positive territory into negative territory</p><p>**AND**</p><p>2- RSI signal went from above 76.5 to below 76.5 (overbought) at least once in the past 28 days **OR** %B signal went from above 1 to below 1 (price broke back into the upper band) at least twice in the past 60-days.</p>|
| - |
- **Hold** otherwise

For Long(+1), short(-1) and hold(0) decisions, I decided to take a more straight forward (all-in) approach, where I either buy all I can or Sell all I can while maintaining the allowable positions of [-1000, 0, 1000].

The reason I combine the indicators in the following way was to minimize the number of false positive signals that may arise from relying on only one indicator for making the decisions. **I believe that my strategy is effective** because it relies on PPO signal as the main buy/sell signal, however, to minimize false signals I also require that RSI or %B indicators to have also given a signal within the past 28 and 60 days respectively. Therefore, in my strategy I combined a trend following approach (using PPO), and an overbought/oversold approach (using RSI and %B).

Figure 1 below shows JPM stock price chart and the 3 technical indicators used as well as the generated buy/sell signals on the in-sample period from 01-01-2008 to 31-12-2009.

1. Showing JPM stock Price and 3 technical indicators used on in-sample period as well as the generated buy(green)/sell(red) signals in vertical lines.

To develop and validate my strategy I back tested it against a benchmark buy and hold strategy. Figure 2 below shows that my strategy **outperformed** the benchmark strategy generating a cumulative return of 0.2288 vs -0.0379 for benchmark for **in-sample** period.

1. Manual strategy vs benchmark for JPM **on in-sample** period in addition to buy(blue)/sell(black) signals.

To validate that my strategy works I tested it on an out-of-sample period from 01-01-2010 to 31-12-2011. Figure 3 below validates that my strategy works well and still **outperforms** the benchmark with cumulative return of 0.1185 vs -0.1337 for benchmark over the **out-of-sample** period.

1. Manual strategy vs benchmark for JPM on **out-of-sample** period in addition to buy(blue)/sell(black) signals.

To simulate the performance and compute the portfolio values I consider a commission fee 0f 9.95$ per transaction and a market impact of -0.005\*price. Table below summarizes the performance of my manual strategy vs the benchmark for JPM stock for in and out-of-sample periods.

<table><tr><th colspan="2" valign="top"></th><th valign="top">Manual Strategy</th><th valign="top">Benchmark</th></tr>
<tr><td rowspan="3" valign="top">In-sample</td><td valign="top">Cumulative Return</td><td valign="top">0\.2288</td><td valign="top">-0.0379</td></tr>
<tr><td valign="top">Standard deviation daily return</td><td valign="top">0\.0139</td><td valign="top">0\.0175</td></tr>
<tr><td valign="top">Average daily return</td><td valign="top">0\.0005</td><td valign="top">7\.507e-05</td></tr>
<tr><td rowspan="3" valign="top">Out-of-sample</td><td valign="top">Cumulative Return</td><td valign="top">0\.1185</td><td valign="top">-0.1337</td></tr>
<tr><td valign="top">Standard deviation daily return</td><td valign="top">0\.0071</td><td valign="top">0\.0089</td></tr>
<tr><td valign="top">Average daily return</td><td valign="top">0\.00025</td><td valign="top">-0.00025</td></tr>
</table>

From the table above, we could observe that manual strategy beats the benchmark with a big margin over both the in-sample and out of sample periods. However, as expected the cumulative return and average daily return over the in-sample period is higher than out of sample period. Also, worth noting that my strategy is very robust such that the standard deviation of daily return in my out of sample period is less than that of in-sample period and less than benchmark in both datasets. The difference in returns between in sample and out sample is because JPM behaved in a relatively different way. We could see from benchmark that JPM in general performed worse in out of sample period compared to in sample. In addition, my strategy generated fewer signals for the out of sample timeframe, hence less trades and hence less potential for profits. 
1. # **STRATEGY LEARNER**
My Strategy Learner idea is to train a model to learn how to predict if the price  **N=14** days in the future is going be higher/lower than current price. By training this model on training data, we can use it to predict if the N day return is going to be positive (Long) or negative(short) on new data.  For My strategy learner I utilized the Classification-based learner that uses a RandomTree Learner with Bagging. My RandomTree learner algorithm starts by randomly picking and an attribute to split on from the training data, then finds the best split value (median) for this attribute and continues to split until each leaf node has at most **5 data points** (leaf-size hyperparameter). This value for leaf-size was chosen after performing roll forward cross-validation on the in-sample data. Where I split the data into 10 slices, trained model on 1 slice and tested on subsequent slice and so on (code in strategyLearner.py). Finally, I developed a bagging algorithm to take my weak learners and run them on different randomly selected subsets of the data and then taking the prediction as the mode prediction of all learners. RandomTree learner is generally a weak learner and tends to suffer from over-fitting. However, with bagging, which combines many week learners trained on subsets of the data, I could reduce the effect of overfitting and therefore enhance model performance. Overfitting is when the model memorizes the training data but performs much worse on future data.

To prepare my training data that will be the input into my strategy learner, I created a single dataframe made of 3 columns namely: PPO, RSI & %B indexed by date in ascending order. Therefore, my training data is a timeseries data showing the value of each of my 3 technical indicators per day. In addition, I also Normalized the values so that they are using similar scales to avoid one indicator like RSI to take over from the rest. To prepare the predictions column for the classifier I first calculated the N-day return by comparing the price of each day with the price of **N=14** days in advance. I also subtract the commission and market impact to account for their effects. I then filled the predictions column (Y) with values (1,0,-1) based on the below algorithm:

|<p>N\_day\_ret = (price[t+14]/price[t]) – 1.0 – commission – impact\*price[t]</p><p>if ret > 0.025:</p><p>`        `Y[t] = +1 # LONG</p><p>else if ret < -0.025:</p><p>`        `Y[t] = -1 # SHORT</p><p>else:</p><p>`        `Y[t] = 0 # HOLD</p>|
| :- |

I choose 14 days as my N day period because all my indicators have a lookback period of 14 days so looking 14 days in advance gives the indicators the ability to catchup.  Also, I selected 0.025 and -0,025 as my thresholds for Long and short by observing the average change in daily returns and iteratively try different values so that I’m reducing false positive signals. Similar to manual strategy, For Long and short decisions, I decided to take a more straight forward (all-in) approach, where I either buy all I can or Sell all I can while maintaining the allowable positions of [-1000, 0, 1000].
1. # **EXPERIMENT 1**
This experiment is designed to compare the performance of strategy learner against Manual Strategy and a benchmark (buy and hold strategy) over in-sample period of 01-01-2008 to 31-12-2009 for JPM stock. The performance is compared based on normalized returns, where the daily return is normalized by dividing by the price of the first day, so that all prices start from 1. 

First, I compute the benchmark and manual strategy trading signals and returns as per section 3 and then strategy learner trading signals is also computed as discussed in section 4 for the in-sample period. To simulate the performance and compute the portfolio values I consider a commission fee 0f 9.95$ per transaction and a market impact of -0.005\*price.  The results are finally plotted below on the same chart (figure 4) to compare performance.

1. Normalized returns for Manual Strategy vs Strategy learner vs benchmark on in-sample period for JPM

As hypothesized, the **strategy learner out performs** both the manual strategy and benchmark over the in-sample period. The model is looking at the indicators and then for example says “based on what the indicators values are today, we are observing a lower 14-day future return and thus we will predict best action is to "be short (-1)". This added value is only over the in-sample period however and must be tested over the out of sample period for confirmation on model effectiveness. I also expect strategy learner to beat benchmark if I repeat the experiment under the same conditions, because the strategy learner is by default looking at a finer level of detail than my manual learner and benchmark, and therefore can tease out those finer details and make more higher resolution decisions. 
1. # **EXPERIMENT 2**
This experiment is designed to evaluate how changing the value of impact should affect in-sample trading behavior. The hypothesis is that that if we train strategy Lerner to expect high impact, more likely we are going to make more conservative trades and therefore less buy/sell signals and hence missing out on potential returns.

To test this hypothesis, I choose two approaches. One is to look at the how the Normalized daily returns for JPM stock look like over the in-sample period for 3 different impact levels [0.0, 0.05, 0.005]. Figure 5 below shows the results.

1. Normalized Returns for JPM under different impact levels over in-sample period

From the figure above, we could see that the higher the impact the less trading opportunities are utilized the lower the returns. For this experiment the commission and market impact levels used in market simulator to compute portfolio values are 9.95$ per trade and 0.005 respectively.

For the second approach I used 2 metrics to measure my portfolio performance over in-sample period namely cumulative return and sharp ratio for the same 3 impact levels [0.0, 0.05, 0.005]. Figure 6 below shows the results.

1. Cumulative returns and sharp ratio for JPM stock over the in-sample period for 3 different impact levels

The results from the second approach also show that cumulative return decreases as impact increases due to being conservative and loosing many trading opportunities. Another observation is that the risk adjusted reward ratio also goes down as the model is trained over higher impact levels this could potentially be due to the lower return we get with higher impact levels.
12

