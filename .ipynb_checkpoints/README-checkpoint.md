# fintech_project_two
![Image](https://raw.githubusercontent.com/filprager/fintech_project_two/layout/image/Etm4yFZUcAAoN5u.jpeg)

## How to run the app
`python3 main.py`

## Hypothesis
Silicon Valley tech entrepreneurs tweets (e.g. from Elon Musk, Jack Dorsey etc) have a high correlation to stock/crypto price movements, and that this movement is relatively short term (< 1day)

## Goal
A web app (or Chatbot - stretch goal) where a user can choose to analyse the historical stock/crypto price correlation to what a selected tech entrepreneur has said recently, and get an auto trading strategy based on it.

## Input
* Tweets from Twitter API
* Keyword searches from Google Analytics API (stretch goal)
* Stock/crypto prices and volumes from Yahoo Finance API (minimum hourly data)
* Entrepreneur’s name from a drop-down list (minimum Elon Musk, Jack Dorsey, Mark Cuban)
* Stock/crypto name and/or ticker (e.g. Bitcoin, BTC)
* Specific time period (start/end date) of the stock/crypto and twitter data.  Alternatively have preset period

## Output
* Keyword, synonyms and associated sentiment of what the selected entrepreneur has said recently in tweets 
    - Present as Wordcloud
* Historical price curve of the chosen stock/crypto
* Correlation curve between stock/crypto price and tweet sentiment, over time
* Comparison of different models (e.g. Random Forest vs. Neural Network)
* An algo trading strategy
    - A plot showing long/short signals
    - A plot showing cumulative returns
    - A table showing pottfolio metrics


## Work distribution
* Set up a layout for the code (main.py, dashboard) - Mark
* Fetch data (APIs) + Clean data  - Marianna
* Process data (including machine learning) -  3-4 people
    - Wordcloud
    - Key words count
    - Sentiment analysis
    - Correlation analysis
    - Algorithmic trading
        - Random Forest
        - Neural Network
* Migrate to AWS - TBC

## Key Dates
* Draft code (mostly working) - 28th February, Sunday
* Final code (all bugs addressed) - 5th March, Thursday
* Final presentation - 6th March, Saturday

fils test - this should be the only change

