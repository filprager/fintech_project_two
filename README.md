#FINTECH PROJECT TWO
![Image](https://raw.githubusercontent.com/filprager/fintech_project_two/layout/image/Etm4yFZUcAAoN5u.jpeg)


# SillyCon

## How to run the app

`python3 main.py`

## Introduction
At the inception of fintech project two,the bitcoin and dogecoin prices skyrocketed after Elon Musk's Tweeted about them.Conincidentally,the price of Gamestop was causing trouble in Wallstreet,after reddit users manipulated the stock price of Gamestop. As such,we thought that there is a correlation between the tweets of Elon Musk and the price movement of bitcoin and dogecoin.This is how the idea for the project came about.


## Hypothesis
Silicon Valley tech entrepreneurs tweets (e.g. from Elon Musk, Jack Dorsey etc) have a high correlation to stock/crypto price movements, and that this movement is relatively short term (< 1day)


## Goal
A web app (or Chatbot)  where a user can choose to analyse the historical stock/crypto price correlation to what a selected tech entrepreneur has said recently, and get an auto trading strategy based on it.

## Installation requirements

`pandas` 

`pathlib`

`hvplot`

`tensor flow`

`sklearn`

`numpy`


## Workflow

- Fetch data ( Tweets from Twitter API ,Stock/crypto prices and volumes from Yahoo Finance API (minimum hourly data),and Specific time period (start/end date) of the stock/crypto and twitter data or have preset period )  + Clean data  
- Process data (including machine learning) and layout for the code(main.py, dashboard)-
    - Wordcloud
    - Key words count
    - Sentiment analysis
    - Correlation analysis
    - Algorithmic trading
        - Random Forest
        - Neural Network
    -dashboard( Entrepreneur’s name from a drop-down list and Stock/crypto name and/or ticker (e.g. Bitcoin, BTC) )  
- Migrate to AWS - TBC


## APP Output
- Present as Wordcloud
- Historical price curve of the chosen stock/crypto
- Comparison of different models (e.g. Random Forest vs. Neural Network)
- An algo trading strategy
    - A plot showing long/short signals
    - A plot showing cumulative returns
    - A table showing pottfolio metrics



