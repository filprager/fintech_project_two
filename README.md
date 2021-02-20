# fintech_project_two

## Hypothesis / Purpose
Silicon Valley tech entrepreneurs tweets (e.g. from Elon Musk, Jack Dorsey etc) have a high correlation to stock/crypto price movements, and that this movement is relatively short term (< 1day)

## Features
A Notebook (or Amazon Lex Chatbot - stretch goal) where a user can choose to analyse the historical stock/crypto price correlation to Twitter sentiment from a selected tech entrepreneur

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
* A website/dashboard/download report providing a more detailed analysis (stretch goal)

## Work distribution
* Design the structure of the code (main.py) + Visualise data (dashboard) - Mark
* Fetch data (APIs)  - Marianna
* Clean data + Process data (including machine learning) -  2-3 people
* Migrate to AWS - TBC

## Key Dates
* Draft code (mostly working) - 28th February, Sunday
* Final code (all bugs addressed) - 5th March, Thursday
* Final presentation - 6th March, Saturday

