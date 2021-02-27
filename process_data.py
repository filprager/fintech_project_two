import pandas as pd
import hvplot.pandas

# Run all functions below and return their plots
def process_data():
    '''
    Returns: HvPlot object
    '''
    
    # Run all functions 
    tweeting_price_curve_doge, tweeting_price_curve_btc = make_tweeting_price_curve()
    cumulative_doge_curve, cumulative_bitcoin_curve = make_cumulative_curve()
    bitcoin_price_curve, dogecoin_price_curve = make_price_curve()
    # @TODO: Algo trading based on Random Forest Tree
    # @TODO: Algo trading based on RNN

    # Return all plots
    return (
            tweeting_price_curve_doge,
            tweeting_price_curve_btc,
            cumulative_doge_curve, 
            cumulative_bitcoin_curve,
            bitcoin_price_curve,
            dogecoin_price_curve
            )

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

       
    return cumulative_doge_curve,cumulative_bitcoin_curve


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

