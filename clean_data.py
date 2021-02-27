# Fetch the selected entrepreneur's tweets and crypto price, 
 
# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
# Import libraries
import pandas as pd
import numpy as np


# %%
# Load Elon Musk's tweets
tweet_elon = pd.read_pickle('retrieve/twitter/TWEETDUMP/twitter_elon_all_df')
tweet_elon


# %%
# Load all cryptos' prices
price_all = pd.read_pickle('retrieve/cryptocompare/CRYPTODUMP/hourlyhist_crypto_compare_api_v1_df')
price_all


# %%
# Slice Bitcoin's data out of all cryptos
price_btc = price_all[price_all['coin_symbol']=='BTC']
price_btc


# %%
# Reorder Bitcoin's data from the oldest to the neweset
price_btc = price_btc.sort_values('time_convert')
price_btc


# %%
# Slice Dogecoin's data out of all cryptos
price_doge = price_all[price_all['coin_symbol']=='DOGE']
price_doge


# %%
price_doge[['time_convert', 'close']].sort_values('time_convert').set_index('time_convert')


# %%
# Reorder Dogecoin's data from the oldest to the neweset
price_doge = price_doge.sort_values('time_convert')
price_doge


# %%
# Slice only time and price from  Bitcoin' data
price_btc = price_btc[['time_convert', 'close']]
price_btc


# %%
# Slice only time and price from  Dogecoin' data
price_doge = price_doge[['time_convert', 'close']]
price_doge


# %%
# Set time as Bitcoin data's index
price_btc = price_btc.set_index('time_convert')
price_btc


# %%
# Set time as Dogecoin data's index
price_doge = price_doge.set_index('time_convert')
price_doge


# %%
# Rename Bitcoin's price column to Bitcoin Price
price_btc = price_btc.rename(columns={'close':'Bitcoin Price'})
price_btc


# %%
# Rename Dogecoin's price column to Dogecoin Price
price_doge = price_doge.rename(columns={'close':'Dogecoin Price'})
price_doge


# %%
price_btc.index[0]


# %%
# Slice only time and text from 
tweet_elon = tweet_elon[['created_at', 'text']]
tweet_elon


# %%
tweet_elon.iloc[0, 0][:13]


# %%
# Change the granularity of tweet time to hour  
time_all = []

for index, row in tweet_elon.iterrows():
    time = row['created_at'][:13]
    time_all.append(time)
    
tweet_elon['time in hour'] = time_all
tweet_elon    


# %%
# Convert time from string to pandas Timestamp
tweet_elon['time in hour'] = pd.to_datetime(tweet_elon['time in hour'])


# %%
tweet_elon.iloc[0, 2]


# %%
# Set index to time
tweet_elon = tweet_elon.set_index('time in hour')


# %%
# Remove the created_at column
tweet_elon = tweet_elon.drop(columns='created_at')
tweet_elon


# %%
# Create a new DataFrame that will combine tweets and cryptos' prices
df_combined = price_btc.copy()
df_combined


# %%
# Add Dogecoin price to the combined DataFrame
df_combined = df_combined.join(price_doge, how='inner')
df_combined


# %%
# Remove NaN
df_combined = df_combined.dropna()
df_combined


# %%
# Slice data from 2020
df_combined = df_combined.loc['2020':]
df_combined


# %%
# Create a Tweet column, set initial value as an emppty list
df_combined["Elon Musk's Tweet"] = np.empty((len(df_combined), 0)).tolist()
df_combined


# %%
tweet_elon


# %%
# Add Elon Musk's tweets to the combined DataFrame
for index, row in tweet_elon.iterrows():
    if index in df_combined.index:
        df_combined.loc[index, "Elon Musk's Tweet"].append(row['text'])

df_combined        


# %%
# Change the name of index to Time
df_combined.index = df_combined.index.rename('Time')
df_combined


# %%
# Slice a DataFrame that contains only Elon Musk's tweet and Dogecoin price
elon_doge = df_combined[["Elon Musk's Tweet", "Dogecoin Price"]]
elon_doge


# %%
# Slice a DataFrame that contains only Elon Musk's tweet and Bitcoin price
elon_btc = df_combined[["Elon Musk's Tweet", "Bitcoin Price"]]
elon_btc = elon_btc.dropna()
elon_btc


# %%
# For the Bitcoin DataFrame, rename the  tweet column to show the format of the tweets
elon_btc = elon_btc.rename(columns={"Elon Musk's Tweet":"Elon Musk's Tweet in List"})
elon_btc


# %%
# For the Dogecoin DataFrame, rename the tweet column to show the format of the tweets
elon_doge = elon_doge.rename(columns={"Elon Musk's Tweet":"Elon Musk's Tweet in List"})
elon_doge


# %%
# For the Bitcoin DataFrame, add a new column that contains tweets in string
elon_btc["Elon Musk's Tweet in String"] = [','.join(map(str, l)) for l in elon_btc["Elon Musk's Tweet in List"]]
elon_btc


# %%
# For the Dogecoin DataFrame, add a new column that contains tweets in string
elon_doge["Elon Musk's Tweet in String"] = [','.join(map(str, l)) for l in elon_doge["Elon Musk's Tweet in List"]]
elon_doge


# %%
# Add a new column that contains only tweets that mention the word Bitcoin (not case sensitive)
elon_btc["Elon Musk's Tweet That Mentions the Word Bitcoin"] = (elon_btc[elon_btc["Elon Musk's Tweet in String"].str.contains('bitcoin', case=False)]["Elon Musk's Tweet in String"])
elon_btc


# %%
# Add a new column that contains only tweets that mention the word DOGE (not case sensitive)
elon_doge["Elon Musk's Tweet That Mentions the Word DOGE"] = (elon_doge[elon_doge["Elon Musk's Tweet in String"].str.contains('doge', case=False)]["Elon Musk's Tweet in String"])
elon_doge


# %%
# Add a new column that contains only tweets that mention the word BTC (not case sensitive)
elon_btc["Elon Musk's Tweet That Mentions the Word BTC"] = (elon_btc[elon_btc["Elon Musk's Tweet in String"].str.contains('btc', case=False)]["Elon Musk's Tweet in String"])
elon_btc


# %%
# Add a new column that contains tweets that mention the word Bitcoin or BTC
elon_btc["Elon Musk's Tweet That Mentions the Word Bitcoin or BTC"] = elon_btc["Elon Musk's Tweet That Mentions the Word Bitcoin"].astype(str) + elon_btc["Elon Musk's Tweet That Mentions the Word BTC"].astype(str)
elon_btc


# %%
# Add a column that contain 0 and 1, 0 if the tweet doesn't mention neither Bitcoin or BTC, 1 if one of them gets mentioned
list_temporary = []
for index, row in elon_btc.iterrows():
    if row["Elon Musk's Tweet That Mentions the Word Bitcoin or BTC"] == 'nannan':
        list_temporary.append(0)
    else:
        list_temporary.append(1)
        
elon_btc["Does Elon Musk's Tweet Tention the Word Bitcoin or BTC?"] = list_temporary       
    
elon_btc  


# %%
# Add a column that contain 0 and 1, 0 if the tweet doesn't mention DOGE, 1 if it does
elon_doge["Does Elon Musk's Tweet Tention the Word DOGE?"] = elon_doge["Elon Musk's Tweet in String"].str.contains('doge', case=False).astype(int)
elon_doge


# Save the DataFrame into a pickle file
elon_doge.to_pickle('./data/elon_doge.plk')
elon_btc.to_pickle('./data/elon_btc.plk')
