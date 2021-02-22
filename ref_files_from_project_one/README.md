# FinTech Project 1

# DeLorean - Financial Analysis Tool

## View Past and Projected Future Returns for Stocks, with Risk Analysis


### Main APP

#### Installation

pip install datetime
pip unstall pytz
pip install python-dotenv
pip install alpaca-trade-api
pip install package_name
conda install panel
conda install panda
conda install numpy  
 conda install -c anaconda nb_conda -y
conda install -c conda-forge nodejs -y
conda install -c pyviz holoviz -y
conda install -c plotly plotly -y

#### Required Files

main.ipyd
get_data.py
process_data.py
visualize_data.py
make_interface.py
MCForecastTools.py

#### Start Up

Start app by executing 'main.ipyd'

#### WORKFLOW

1. Enter Start Date
2. Enter End Date
3. Add 1-5 Stocks with manual portfolio weights (must sum to 1)
4. Press Run Interact button

Note - Monte Carlo analysis processing takes 3-5 seconds - please wait after pressing Run Interact button

#### Example Schema

portfolio_1{
ticker{
"MSFT" : 0.2,
"AAPL": 0.2,
"TSLA": 0.2,
"GOOG": 0.2,
"BRK": 0.2  
 },
date_range{
start_date: 11/12/2019,
end_data: 11/12/2020
}
}

#### Data used

- User inputs
- Alpaca API
- Monte Carlo

#### APP Output

In a TAB layout:

- Past Performance of selected stocks, combined portfolio and the market (S&P500) - Various Daily Returns, Cumulative Returns
- Future Performance of the combined portfolio - Monte Carlo simulation
- Risk Analysis of selected stocks - Box plot, Correlation graph, Rolling Standard Deviation of Daily Returns over a 21 day window

### Future Improvements

1. Refine ticker counts and granularity weights
2. Migrate to a new finance API
3. Update Monte Carlo py
4. Create and compare multiple custom portfolios
5. Integration with exchanges to offer 'Buy Now' functionality for selected stocks
6. Improved GUI for Web and Mobile - connect front-end UI to Python for processing


### Flask UI App (Work in Progress)

#### Installation

pip install flask
pip install flask_wtf
pip install wtforms
pip install wtforms.validators

#### Start Up

Start app by executing 'flaskinput.py'

#### Required Files

flaskinput.py
ui_output.py
/assets
/templates