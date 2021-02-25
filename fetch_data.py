# Fetch the selected entrepreneur's words and crypto's price, wrap them into a DataFrame and return it 
def fetch_data(entrepreneur, crypto):
    '''
    Parameters
    ---------- 
    entrepreneur: str
    crypto: str  

    Returns
    -------
    Pandas DataFrame    
    '''

    
    # Import libraries
    import pandas as pd
    import numpy as np

    # Load Elon Musk's tweets
    tweet_elon = pd.read_pickle('./retrieve/twitter/TWEETDUMP/twitter_elon_all_df')

    # Load all cryptos' prices
    price_all = pd.read_pickle('./retrieve/cryptocompare/CRYPTODUMP/hourlyhist_crypto_compare_api_v1_df')

    # Slice Bitcoin's data out of all cryptos
    price_btc = price_all[price_all['coin_symbol']=='BTC']

    # Reorder Bitcoin's data from the oldest to the neweset
    price_btc = price_btc.sort_values('time_convert')

    # Slice Dogecoin's data out of all cryptos
    price_doge = price_all[price_all['coin_symbol']=='DOGE']

    # Reorder Dogecoin's data from the oldest to the neweset
    price_doge = price_doge.sort_values('time_convert')

    # Slice only time and price from  Bitcoin' data
    price_btc = price_btc[['time_convert', 'close']]

    # Slice only time and price from  Dogecoin' data
    price_doge = price_doge[['time_convert', 'close']]

    # Set time as Bitcoin data's index
    price_btc = price_btc.set_index('time_convert')
    price_btc

    # Set time as Dogecoin data's index
    price_doge = price_doge.set_index('time_convert')

    # Rename Bitcoin's price column to Bitcoin Price
    price_btc = price_btc.rename(columns={'close':'Bitcoin Price'})

    # Rename Dogecoin's price column to Dogecoin Price
    price_doge = price_doge.rename(columns={'close':'Dogecoin Price'})

    # Slice only time and text from 
    tweet_elon = tweet_elon[['created_at', 'text']]

    # Change the granularity of time to house  
    time_all = []

    for index, row in tweet_elon.iterrows():
        time = row['created_at'][:13]
        time_all.append(time)
        
    tweet_elon['time in hour'] = time_all 


    # Convert time from string to pandas Timestamp
    tweet_elon['time in hour'] = pd.to_datetime(tweet_elon['time in hour'])

    # Set index to time
    tweet_elon = tweet_elon.set_index('time in hour')

    # Remove the created_at column
    tweet_elon = tweet_elon.drop(columns='created_at')

    # Create a new DataFrame that will combine tweets and cryptos' prices
    df_combined = price_btc.copy()

    # Add Dogecoin price to the combined DataFrame
    df_combined = df_combined.join(price_doge, how='outer')

    # Create a Tweet column, set initial value as an empty list
    df_combined["Elon Musk's Tweet"] = np.empty((len(df_combined), 0)).tolist()

    # Add Elon Musk's tweets to the combined DataFrame
    for index, row in tweet_elon.iterrows():
        if index in df_combined.index:
            df_combined.loc[index, "Elon Musk's Tweet"].append(row['text'])

      

    # Change the name of index to Time
    df_combined.index = df_combined.index.rename('Time')


    # Slice a DataFrame that contains only Elon Musk's tweet and Dogecoin price
    elon_doge = df_combined[["Elon Musk's Tweet", "Dogecoin Price"]]


    # Slice a DataFrame that contains only Elon Musk's tweet and Bitcoin price
    elon_btc = df_combined[["Elon Musk's Tweet", "Bitcoin Price"]]
    elon_btc = elon_btc.dropna()
    
# entrepreneur, crypto

    if entrepreneur == 'Elon Musk' and crypto == 'Bitcoin':
        return elon_btc
    elif entrepreneur == 'Elon Musk' and crypto == 'Dogecoin':  
        return elon_doge
    else: 
        return 'Only Elon Musk, Bitcoin and Dogecoin are valid parameters'    

