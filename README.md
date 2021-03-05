
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


## User Input


- A drop down list consisting two selections :


     - Elon Musk Tweets

     - Google Trends Data

- A drop down list for choosing the cryptos given:


     - Bitcoin

     - Dogecoin

## APP Output

- A line plot showing Bitcoin's historical price with some markers on the top indicating when Elon Musk tweets something about Bitcoin

- A line plot showing Dogecoin's historical price with some markers on the top indicating when Elon Musk tweets something about Dogecoin

-  A line plot showing Bitcoin's cumulative returns 

- A line plot showing Dogecoin's cumulative returns 

- A line plot showing Bitcoin's historical price

- A line plot showing Dogecoin's historical price 

- Plots that shows results of algorithmic trading based on   Random Forest (price only, no tweets)

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




## libraries Used

`pandas` , `pathlib` , `hvplot` , `tensorflow` , `sklearn` , `dotenv` , `numpy` , `random ` , `os` , `json` , `pickle`  , `re` , `time` , `bs4 ` , `urllib` , `requests` , `datetime` , `sys` , `collections`



## APIs Used

`Twitter API `,  `Cryptocompare API`