# Import libraries
from panel.interact import interact_manual
import panel as pn

# Import self-made functions
from build_dashboard import build_dashboard
from process_data import make_tweeting_price_curve, make_cumulative_curve, make_price_curve, make_random_forest

# Analyse the correlation between the chosen entrepreneur and crypto
def analyse(Source, Ticker):
    
    ##Perform user input checks
    # Check if user has selected any input
    if Source == 'Select Source' and Ticker == 'Select Ticker':
        return "Make sure to select both a source and ticker!"
    
    # Check if user has selected source
    if Source == 'Select Source':
        return "Make sure to select a source too!"
    
    # Check if user has selected ticker
    if Ticker == 'Select Ticker':
        return "Make sure to select a crypto ticker too!"
    
    
    ## Process the data and return plots
    # xxx
    tweeting_price_curve_doge, tweeting_price_curve_btc = make_tweeting_price_curve()
    
    # xxx
    cumulative_return_curve_doge,cumulative_return_curve_btc = make_cumulative_curve()
    
    # xxx
    price_curve_btc, price_curve_doge = make_price_curve()
    
    # xxx
    rf_ema_closing_prices, rf_ema_daily_return_volatility, rf_plot3, rf_plot4, rf_plot5, rf_plot6, rf_plot7 = make_random_forest()

    
    ## Create a dashboard to visualise the plots
    dashboard = build_dashboard(
                                tweeting_price_curve_doge, 
                                tweeting_price_curve_btc, 
                                cumulative_return_curve_doge, 
                                cumulative_return_curve_btc, 
                                price_curve_btc, 
                                price_curve_doge, 
                                rf_ema_closing_prices,
                                rf_ema_daily_return_volatility,
                                rf_plot3,
                                rf_plot4,
                                rf_plot5,
                                rf_plot6,
                                rf_plot7
                                # Add more plots here, remember to match the build_dashboard file too
                                )

    ## Return the dashboard
    return dashboard


# Make interactive drop-down lists which users can choose an entrepreneur and stock/crypto from
user_input = interact_manual(
                                analyse, 
                                Source=['Select Source', 'Elon Musk Tweets', 'Google Search Data'], 
                                Ticker=['Select Ticker', 'Bitcoin', 'Dogecoin']
                            )

# Make the interface for the app
interface = pn.Column(
                        '# SillyCon App',
                        '## Analyse correlations between Social Sources and Crypto Prices',
                        '### Select a Source and Crypto Ticker, then press Run Interact', 
                        user_input
                    )

# Launch the app in browser
interface.show()