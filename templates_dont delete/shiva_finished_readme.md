# fintech_project_two
![Image](https://raw.githubusercontent.com/filprager/fintech_project_two/layout/image/Etm4yFZUcAAoN5u.jpeg)

## Main APP 
SillyCon

## Introduction
At the inception of fintech project two,the bitcoin and dogecoin prices skyrocketed after Elon Musk's Tweeted about them.Conincidentally,the price of Gamestop was causing trouble in Wallstreet,after reddit users manipulated the stock price of Gamestop.As such,we thought that there is a correlation between the tweets of Elon Musk and the price movement of bitcoin and dogecoin.This is how the idea for the project came about.


## Hypothesis
Silicon Valley tech entrepreneurs tweets (e.g. from Elon Musk, Jack Dorsey etc) have a high correlation to stock/crypto price movements, and that this movement is relatively short term (< 1day)


## Goal
A web app (or Chatbot)  where a user can choose to analyse the historical stock/crypto price correlation to what a selected tech entrepreneur has said recently, and get an auto trading strategy based on it.

## Import libraries and dependencies
'from panel.interact import interact_manual'
'import panel as pn'
'import hvplot.pandas'
'import numpy as np'
'import pandas as pd'
'from pathlib import Path'
'import hvplot.pandas'
'from sklearn.ensemble import RandomForestClassifier'
'from sklearn.datasets import make_classification'


## WORKFLOW 

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


### How to run the app

`python3 main.py`

## APP Output
- Present as Wordcloud
- Historical price curve of the chosen stock/crypto
- Correlation curve between stock/crypto price and tweet sentiment, over time
- Comparison of different models (e.g. Random Forest vs. Neural Network)
- An algo trading strategy
    - A plot showing long/short signals
    - A plot showing cumulative returns
    - A table showing pottfolio metrics


### Future Improvements

1. Increase the data fields interms of entrepreneurs and tweets numbers.
2. not just limited to Twitter rather we can include all possible sources of statements such as publication,journals etc.
3. Create and compare multiple ML models for better trading signals.
4. can use asynco function for faster data processing
5. This app is limited just for predictions but not for advisory purposes, could have gone to that extend. 
6. we can link it to the online trading plateform if a user wants to do so.
7. Interface it with AWS lex and lambda, which will need fair bit of time to accomplish.
