<<<<<<< HEAD
# Create the Random Forest function
def function_random_forest():
=======
import pandas as pd
import hvplot.pandas

# Make a curve that shows tweeting times vs crypto price
def make_tweeting_price_curve():
    '''
    Return: HvPlot object
    '''

    # Load data for Dogecoin
    df_elon_doge = pd.read_pickle('./data/elon_doge.plk')
    df_elon_doge


    # %%
    # Visualize tweeting times for Dogecoin
    tweeting_doge = df_elon_doge[df_elon_doge["Does Elon Musk's Tweet Tention the Word DOGE?"] == 1]['Dogecoin Price'].hvplot.scatter(
        color='royalblue',
        marker='circle',
        size=50,
        legend=False,
        ylabel='Price in $',
        width=1000,
        height=400
    )

    tweeting_doge


    # %%
    # Visualize close price of Dogecoin
    price_curve_doge = df_elon_doge[['Dogecoin Price']].hvplot(
        line_color='lightgray',
        ylabel='Price in $',
        width=1000,
        height=400
    )

    price_curve_doge


    # %%
    # Overlay plots for Dogecoin
    tweeting_price_curve_doge = price_curve_doge * tweeting_doge
    tweeting_price_curve_doge.opts(width=1000, title='When Elon Musk tweets about Dogecoin', ylabel='Dogecoin price in $')


    # %%
    # Load data for Bitcoin
    df_elon_btc = pd.read_pickle('./data/elon_btc.plk')
    df_elon_btc


    # %%
    # Visualize tweeting times for Bitcoin
    tweeting_point_btc = df_elon_btc[df_elon_btc["Does Elon Musk's Tweet Tention the Word Bitcoin or BTC?"] == 1]['Bitcoin Price'].hvplot.scatter(
        color='royalblue',
        marker='circle',
        size=50,
        legend=False,
        ylabel='Price in $',
        width=1000,
        height=400
    )

    tweeting_point_btc


    # %%
    # Visualize close price of Bitcoin
    price_curve_btc = df_elon_btc[['Bitcoin Price']].hvplot(
        line_color='lightgray',
        ylabel='Price in $',
        width=1000,
        height=400
    )



    price_curve_btc


    # %%
    # Overlay plots for Bitcoin
    tweeting_price_curve_btc = price_curve_btc * tweeting_point_btc
    tweeting_price_curve_btc.opts(width=1000, title='When Elon Musk tweets about Bitcoin', ylabel='Bitcoin price in $')

    # Return tweeting vs price curves
    return tweeting_price_curve_doge, tweeting_price_curve_btc


def make_cumulative_curve():                   
    df1= pd.read_pickle("data/elon_doge.plk")
    df1.head()

          
    # Drop unnecessary columns
    df1_doge =df1.drop(columns=["Elon Musk's Tweet in List", 
                                "Elon Musk's Tweet in String",
                                "Elon Musk's Tweet That Mentions the Word DOGE",
                                "Does Elon Musk's Tweet Tention the Word DOGE?"])
    df1_doge.head()
           
    # Calculate the daily return using the 'shift()' function
    daily_returns = (df1_doge - df1_doge.shift(1)) / df1_doge.shift(1)
    daily_returns.head()

        
    # Calculate the cumulative returns using the 'cumprod()' function
    cumulative_returns = (1 + daily_returns).cumprod()-1
    cumulative_returns.head()
    
    # Plot the daily returns of the S&P 500 over the last 5 years
    cumulative_doge_curve =cumulative_returns.hvplot(figsize=(10,5))
    
    
    df2=pd.read_pickle("data/elon_btc.plk")
    df2.head()


    # Drop unnecessary columns
    df2_bit =df2.drop(columns=["Elon Musk's Tweet in List",
                               "Elon Musk's Tweet in String",                               
                               "Elon Musk's Tweet That Mentions the Word Bitcoin",
                               "Elon Musk's Tweet That Mentions the Word BTC",
                               "Elon Musk's Tweet That Mentions the Word Bitcoin or BTC",
                               "Does Elon Musk's Tweet Tention the Word Bitcoin or BTC?"])
    df2_bit.head()
 

    # Calculate the daily return using the 'shift()' function
    daily_returns = (df2_bit - df2_bit.shift(1)) / df2_bit.shift(1)
    daily_returns.head()


    #Calculate the cumulative returns using the 'cumprod()' function
    cumulative_returns = (1 + daily_returns).cumprod()-1
    cumulative_returns.head()


    # Plot the daily returns of the S&P 500 over the last 2 years
    cumulative_bitcoin_curve =cumulative_returns.hvplot(figsize=(10,5))

       
    return cumulative_doge_curve ,cumulative_bitcoin_curve


def make_price_curve():
     
   
    df2=pd.read_pickle("data/elon_btc.plk")
    df2.head()

    df2 =df2.rename(columns={"Bitcoin Price": "Bitcoin_Price"})
    df2.head()
  
    #Historical price curve of the chosen bitcoin
    
    bitcoin_price_curve = df2.Bitcoin_Price.hvplot()
     
 
    df1= pd.read_pickle("data/elon_doge.plk")

    df1.head()
  
    df1 =df1.rename(columns={"Dogecoin Price": "Dogecoin_price"})
    df1

    #Historical price curve of the chosen stock/crypto- ST

    dogecoin_price_curve=df1.Dogecoin_price.hvplot()


    return bitcoin_price_curve, dogecoin_price_curve


def function1():
>>>>>>> main
    plot1 = None
    plot2 = None
    plot3 = None
    plot4 = None
    plot5 = None
    plot6 = None
    plot7 = None
    table1 = None 
    table2 = None 
    table3 = None

    # @TODO: return 1. A wordcloud of what the entrepreneur has said 
    #               2. An HvPlot plot of the historical price curve of the chosen stock/crypto
    #               3. An HvPlot plot of the correlation curve between stock/crypto price and tweet sentiment, over time
    #               4. An HvPlot table of the performance matrix of model1
    #               5. An HvPlot table of the performance matrix of model2
    #               6. An HvPlot showing the long/short signals of the trading strategy
    #               7. An HvPlot showing the lcumulative returns of the trading strategy
    #               8. An HvPlot table showing the pottfolio metrics of the trading strategy
    
    
<<<<<<< HEAD
    # Import libraries and dependencies
    import pandas as pd
    import numpy as np
    from pathlib import Path
#     %matplotlib inline

#     import warnings
#     warnings.filterwarnings('ignore')


    # Set path to Pickle and read in data
    trading_signals_df= pd.read_pickle("data/elon_btc.plk")

    # Drop NAs and calculate daily percent return
    trading_signals_df['daily_return'] = trading_signals_df['Bitcoin Price'].dropna().pct_change()
    
    # Set short and long windows
    short_window = 1
    long_window = 10

    # Construct a `Fast` and `Slow` Exponential Moving Average from short and long windows, respectively
    trading_signals_df['fast_close'] = trading_signals_df['Bitcoin Price'].ewm(halflife=short_window).mean()
    trading_signals_df['slow_close'] = trading_signals_df['Bitcoin Price'].ewm(halflife=long_window).mean()

    # Construct a crossover trading signal
    trading_signals_df['crossover_long'] = np.where(trading_signals_df['fast_close'] > trading_signals_df['slow_close'], 1.0, 0.0)
    trading_signals_df['crossover_short'] = np.where(trading_signals_df['fast_close'] < trading_signals_df['slow_close'], -1.0, 0.0)
    trading_signals_df['crossover_signal'] = trading_signals_df['crossover_long'] + trading_signals_df['crossover_short']

    # Plot the EMA of BTC/USD closing prices
    plot1 = trading_signals_df[['Bitcoin Price', 'fast_close', 'slow_close']].plot(figsize=(20,10))
    
    # Set short and long volatility windows
    short_vol_window = 1
    long_vol_window = 10

    # Construct a `Fast` and `Slow` Exponential Moving Average from short and long windows, respectively
    trading_signals_df['fast_vol'] = trading_signals_df['daily_return'].ewm(halflife=short_vol_window).std()
    trading_signals_df['slow_vol'] = trading_signals_df['daily_return'].ewm(halflife=long_vol_window).std()

    # Construct a crossover trading signal
    trading_signals_df['vol_trend_long'] = np.where(trading_signals_df['fast_vol'] < trading_signals_df['slow_vol'], 1.0, 0.0)
    trading_signals_df['vol_trend_short'] = np.where(trading_signals_df['fast_vol'] > trading_signals_df['slow_vol'], -1.0, 0.0) 
    trading_signals_df['vol_trend_signal'] = trading_signals_df['vol_trend_long'] + trading_signals_df['vol_trend_short']

    # Plot the EMA of BTC/USD daily return volatility
    plot2 = trading_signals_df[['fast_vol', 'slow_vol']].plot(figsize=(20,10))
    
    # Set bollinger band window
    bollinger_window = 20

    # Calculate rolling mean and standard deviation
    trading_signals_df['bollinger_mid_band'] = trading_signals_df['Bitcoin Price'].rolling(window=bollinger_window).mean()
    trading_signals_df['bollinger_std'] = trading_signals_df['Bitcoin Price'].rolling(window=20).std()

    # Calculate upper and lowers bands of bollinger band
    trading_signals_df['bollinger_upper_band']  = trading_signals_df['bollinger_mid_band'] + (trading_signals_df['bollinger_std'] * 1)
    trading_signals_df['bollinger_lower_band']  = trading_signals_df['bollinger_mid_band'] - (trading_signals_df['bollinger_std'] * 1)

    # Calculate bollinger band trading signal
    trading_signals_df['bollinger_long'] = np.where(trading_signals_df['Bitcoin Price'] < trading_signals_df['bollinger_lower_band'], 1.0, 0.0)
    trading_signals_df['bollinger_short'] = np.where(trading_signals_df['Bitcoin Price'] > trading_signals_df['bollinger_upper_band'], -1.0, 0.0)
    trading_signals_df['bollinger_signal'] = trading_signals_df['bollinger_long'] + trading_signals_df['bollinger_short']

    # Plot the Bollinger Bands for BTC/USD closing prices
    plot3 = trading_signals_df[['Bitcoin Price','bollinger_mid_band','bollinger_upper_band','bollinger_lower_band']].plot(figsize=(20,10))

    # Set x variable list of features
    x_var_list = ['crossover_signal', 'vol_trend_signal', 'bollinger_signal']

    # Filter by x-variable list
    trading_signals_df[x_var_list].tail()

    # Shift DataFrame values by 1
    trading_signals_df[x_var_list] = trading_signals_df[x_var_list].shift(1)
    trading_signals_df[x_var_list].tail()

    # Drop NAs and replace positive/negative infinity values
    trading_signals_df.dropna(subset=x_var_list, inplace=True)
    trading_signals_df.dropna(subset=['daily_return'], inplace=True)
    trading_signals_df = trading_signals_df.replace([np.inf, -np.inf], np.nan)
    # trading_signals_df.tail()

    # Construct the dependent variable where if daily return is greater than 0, then 1, else, 0.
    trading_signals_df['Positive Return'] = np.where(trading_signals_df['daily_return'] > 0, 1.0, 0.0)
    # trading_signals_df.tail()

    # Construct training start and end dates
    training_start = trading_signals_df.index.min().strftime(format= '%Y-%m-%d')
    training_end = '2021-01-29'

    # Construct testing start and end dates
    testing_start =  '2021-01-30'
    testing_end = trading_signals_df.index.max().strftime(format= '%Y-%m-%d')
    # # Print training and testing start/end dates
    # print(f"Training Start: {training_start}")
    # print(f"Training End: {training_end}")
    # print(f"Testing Start: {testing_start}")
    # print(f"Testing End: {testing_end}")

    # Construct the x train and y train datasets
    x_train = trading_signals_df[x_var_list][training_start:training_end]
    y_train = trading_signals_df['Positive Return'][training_start:training_end]
    # x_train.tail()
    # y_train.tail()

    # Construct the x test and y test datasets
    x_test = trading_signals_df[x_var_list][testing_start:testing_end]
    y_test = trading_signals_df['Positive Return'][testing_start:testing_end]
    # x_test.tail()
    # y_test.tail()

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification

    # Fit a SKLearn linear regression using just the training set (x_train, y_train):
    model = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=0)
    model.fit(x_train, y_train)

    # Make a prediction of "y" values from the x test dataset
    predictions = model.predict(x_test)

    # Assemble actual y data (y_test) with predicted y data (from just above) into two columns in a dataframe:
    results = y_test.to_frame()
    results["Predicted Value"] = predictions

    # Rename 'Positive Return' column to 'Actual Value' to be more descriptive for the plots below
    results.rename(columns={'Positive Return': 'Actual Value'}, inplace=True)

    # Add 'Return' column from trading_signals_df for the plots below
    results = pd.concat([results, trading_signals_df["daily_return"]], axis=1, join="inner")
    # results.tail()

    # Plot predicted results vs. actual results
    plot4 = results[['Actual Value', 'Predicted Value']].plot(figsize=(20,10))

    # Plot last 10 records of predicted vs. actual results
    plot5 = results[['Actual Value', 'Predicted Value']].tail(10).plot()

    # Replace predicted values 0 to -1 to account for shorting
    results['Predicted Value'].replace(0, -1, inplace=True)
    # results.tail()

    # Calculate cumulative return of model and plot the result
    plot6 = (1 + (results['daily_return'] * results['Predicted Value'])).cumprod().plot()

    # Set initial capital allocation
    initial_capital = 100000  ## <- replace this with user entered value!

    # Plot cumulative return of model in terms of capital
    cumulative_return_capital = initial_capital * (1 + (results['daily_return'] * results['Predicted Value'])).cumprod()
    plot7 = cumulative_return_capital.plot()
    
    
    return plot1, plot2, plot3, plot4, plot5, plot6, plot7, table1, table2, table3



# Create the Neural Network function
def function_neural_network():
    plot1 = None
    plot2 = None
    plot3 = None
    table1 = None 
    table2 = None 
    table3 = None

    # @TODO: return 1. A wordcloud of what the entrepreneur has said 
    #               2. An HvPlot plot of the historical price curve of the chosen stock/crypto
    #               3. An HvPlot plot of the correlation curve between stock/crypto price and tweet sentiment, over time
    #               4. An HvPlot table of the performance matrix of model1
    #               5. An HvPlot table of the performance matrix of model2
    #               6. An HvPlot showing the long/short signals of the trading strategy
    #               7. An HvPlot showing the lcumulative returns of the trading strategy
    #               8. An HvPlot table showing the pottfolio metrics of the trading strategy
    

    ##### MARIANNAS STUFF HERE ######
    
    return plot1, plot2, plot3, table1, table2, table3
=======
    return plot1, plot2, plot3, table1, table2, table3

>>>>>>> main
