# Dimensionality-Reduction-to-forecast-stock-market
Forecasting daily stock market return using dimensionality reduction on financial indicators

The goal of this project was to predict the SPDR S&P 500 ETF (SPY) direction.

1) (data_preprocessing.ipyb)

Firstly we chose to focus on 60 financial indicators (SPY returns, EMA, interest rates, Stocks, etc …) between 2003 and 2013. We downloaded this data from multiple sources : Yahoo finance (SPY and Indexes), Federal Reserve Board of Governors (T Bills, interest rates…), Commodities specialized websites. We then applied calculations to get new variables (returns and EMA), and then we focused on cleaning of the data (replacing NaN values, getting rid of outliers).

2) (dimension_reduction_ANN.ipynb)

We then applied dimensionality reduction to the dataset. We used two methods : PCA (principal component analysis) and KPCA (kernel principal component analysis). The PCA projects the data in lower dimensions the principal components of the data. The KPCA is the same idea as PCA but mapping to a nonlinear input data into a higher-dimensional feature space using a nonlinear mapping (kernel function) and then performing PCA in this feature space.

After the dimensionality reduction, we applied a Neural Network to predict if next day’s returns will be positive or negative. The NN consists of an input layer (size depends on the number of PC that we chose), a hidden layer of 10 nodes and an output layer with 2 nodes (that gives the probability of UP and DOWN). We use the 'tanh' and 'softmax' activation functions.

We did this for input of 12 different sizes (ranging from 1 to 60) for both PCA and KPCA decompositons. We then splitted the data into train, test and validation datasets (60/20/20). We obtained good test accuracy (> 0.5 for all input sizes).

3) (paper_strat.ipynb)

For the trading strategy, we inspired ourselves from 'Forecasting daily stock market return using dimensionality reduction, Xiao Zhong & David Enke'.
So we use the dimensionality reducted data and input it into our NN. If we get a result >0.5 for the 'UP' output, then we buy 1 SPY share and sell it at the end of the day. Else we expect the SPY to go down so we buy 1 share of T-bills and sell at the end of the day. There is no shorting and no leverage in our strategy.

We backtested our strategy with the last 377 days of our data. Our signal is output from the NN and everyday we recalculate the PnL. 

We obtained a highest PnL for PCA when using 3 PC (Sharpe Ratio:0.23 ; Max-Drawdown:1.54 ; W-L ratio:3.03) and for KPCA with 34 PC (Sharpe Ratio:0.37 ; Max-Drawdown:1.17 ; W-L ratio:2.01).
