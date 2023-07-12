# [DESCRIPTION] 
Code for a developed manual strategy vs a machine learning based strategy that uses a random forest ML model to make trading decisions vas a benchmark(buy and hold) strategy. This project is to compare and undertand the diffeence between technical indicatiors and ML models for trading

# [Package description]
##This software package contains following files: -

### indicators.py
Code implementing indicators as functions that operate on DataFrames. 

### marketsimcode.py
Accepts a “trades” DataFrame and simulates this trades. 

### ManualStrategy.py
combine the technical indicators to develop a complex trading strategy

### RTLearner.py
code for random classification tree ML model

### BagLearner.py
performs bagging on RTLearner to enhance performance and reduce overfitting

### strategyLearner.py
prepares the trainig data using indicators.py and passes it to baglearner for determining buy sell signals 

### experiment1.py
experiment to compare manualStrategy to StrategyLearner to benchmark over in-sample period

### experiment2.py
experiment to evaluate the effect of impact on strategy learner.

### testproject.py
This file should be considered the entry point to the project.


# [EXECUTION]
PYTHONPATH=../:. python testproject.py


# [OUTPUT]
1- Indicators and price chart for JPM from 2008-01-01 to 2009-12-31.png
2- Indicators and price chart for JPM from 2010-01-01 to 2011-12-31.png (not incl. in report)
3- ManualStareyvsBenchMark for JPM from 2008-01-01 to 2009-12-31.png
4- ManualStareyvsBenchMark for JPM from 2010-01-01 to 2011-12-31.png
5- experiment1.png
6- experiemnt2 comparison.png
7- experiemnt2 benchmark.png
8- manualStrategyVsBenchmark.txt

