
![Image](https://raw.githubusercontent.com/filprager/fintech_project_two/layout/image/Etm4yFZUcAAoN5u.jpeg)


# SillyCon App


## Job description

- Shiva = General Insights (Historical), Word Cloud, ReadMe, Presentation

- Patricia = AWS Lex chatbot, Dashboard Concepts, Presentation

- Marianna = API Data Retrieval (Pickles), Data Processing, Google Trends, Presentation

- Fil = Main App, Data Processing (Random Forest), Dashboard, ReadMe, Presentation

- Mark = Main App, Data Processing (Fixed, RNN, RF), Read Me, Presentation


## How to run the app

Option 1 - Type `python3 main.py` in Terminal to launch the dashboard
Option 2 - Copy main.py file into a Jupyter Lab notebook (ipynb) file and run to launch the dashboard


## Background
Near the inception of Fintech Project Two, Bitcoin and Dogecoin prices skyrocketed after Elon Musk's tweeted about each coin. Similarly, the price of Gamestop was causing trouble on Wallstreet after Reddit users "manipulated" the price of Gamestop (GME) shares. As such, the team theorised that there is a strong correlation between social media and the price movement of Bitcoin and Dogecoin.  The team decided to build an algorithmic trading assistant to analyse this in further detail for crypto traders.


## Hypothesis
Silicon Valley tech entrepreneurs tweets (e.g. from Elon Musk) have a high correlation to crypto price movements, and that this movement can be successfully predicted with machine learning models. 


## Goal
A web app (or Chatbot) where a user can choose to analyse the historical crypto price correlation to what a selected tech entrepreneur has said recently, and get an auto trading strategy based on it.


## User Input

- A drop-down list consisting two source selections:

     - Elon Musk Tweets

     - Google Trends Data
     

- A drop-down list for choosing the desired crypto coin:

     - Bitcoin

     - Dogecoin
     

## App Output

- A line plot showing Bitcoin's historical price with some markers on the top indicating when Elon Musk tweets something about Bitcoin

- A line plot showing Dogecoin's historical price with some markers on the top indicating when Elon Musk tweets something about Dogecoin

- A line plot showing Bitcoin's cumulative returns 

- A line plot showing Dogecoin's cumulative returns 

- A line plot showing Bitcoin's historical price

- A line plot showing Dogecoin's historical price 

- Plots that shows results of algorithmic trading based on Random Forest (price only, no tweets)

    - A line plot showing Bitcoin's historical price with some markers on the top indicating buy/sell actions

    - A line plot showing portfolio value of the Bitcoin investment with some markers on the top indicating buy/sell actions

    - A table showing the evaluation results of the performance of the Bitcoin investment portfolio

    - A line plot showing Dogecoin's historical price with some markers on the top indicating buy/sell actions

    - A line plot showing portfolio value of the Dogecoin investment with some markers on the top indicating buy/sell actions

    - A table showing the evaluation results of the performance of the Dogecoin investment portfolio

- Plots that show the results of an algorithmic trading based on RNN LSTM

    - A line plot showing the algo's predictions on whether Bitcoin price will rise or fall in each hour

    - A line plot showing the algo trading's cumulative returns on Bitcoin 

    - A line plot showing the algo's predictions on whether Dogecoin price will rise or fall in each hour

    - A line plot showing the algo trading's cumulative returns on Dogecoin



## Libraries Used

`pandas` , `pathlib` , `hvplot` , `tensorflow` , `sklearn` , `dotenv` , `numpy` , `random ` , `os` , `json` , `pickle`  , `re` , `time` , `bs4 ` , `urllib` , `requests` , `datetime` , `sys` , `collections`



## APIs Used

`Twitter API `,  `Cryptocompare API`



## Explanation of Each File and Folder 

- 'main.py' = Main file which co-ordinates the entire app and calls functions in the other .py files (used for launching the app)

- 'retrieve' folder (various files) = Fetches raw data from Twitter and Cryptocompare, and creates Pickle files for processing

- 'clean_data.py' = Processes raw data (Pickle files) into suitable clean dataframes for consumption by machine learning models

- 'algo_trading_rf' = Prepares data and trains model for Random Forest (Price + Tweets)

- 'algo_trading_rnn' = Prepares data and trains model for Recurrent Neural Network (Price + Tweets)

- 'model' folder (various files) = Stores RF and RNN models

- 'data' folder (various files) = Stores model datasets / dataframes (e.g. test and train)

- 'process_data.py' = Prepares data and trains model for Random Forest (Price only).  Runs all models and creates plots for Trading Strategies and General Insights

- 'make_word_cloud.py' = Generates WordCloud images for plotting

- 'build_dashboard.py' = Creates dashboard layout for Interact function outputs

- 'chatbot' folder = Contains AWS Chatbot concept model (json)

- 'image' folder (various files) = Contains WordCloud and ReadMe images


