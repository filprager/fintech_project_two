# Import libraries
from panel.interact import interact_manual
import panel as pn

# Import self-made functions
from build_dashboard import (
                                build_dashboard_btc,
                                build_dashboard_doge,
                                build_dashboard_google_trends_btc
                            )

from process_data import (
                            make_tweeting_price_curve, 
                            make_cumulative_curve, 
                            make_price_curve, 
                            make_random_forest, 
                            algo_trading_fixed_strategy, 
                            load_algo_trading_result_rnn, 
                            load_algo_trading_result_rf,
                            load_trading_result_google_search
                        )

# Analyse the correlation between the chosen social source and crypto
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
    
    # Check if user has selected Google Trends and block Dogecoin (not available at present)
    if Source == 'Google Trends Data' and Ticker == 'Dogecoin':
        return "Sorry - Google Trends does not support Dogecoin at present!"
    
    
    ## Process the data and return plots from the various functions
    # Plots for Price Curve with Tweets marked
    (
        tweeting_price_curve_btc, 
        tweeting_price_curve_doge
    ) = make_tweeting_price_curve()
    
    # Plots for Cumulative Curve
    (
        cumulative_return_curve_btc, 
        cumulative_return_curve_doge
    ) = make_cumulative_curve()
    
    # Plots for Price Curve
    (
        price_curve_btc, 
        price_curve_doge
    ) = make_price_curve()
    
    # Plots for Random Forest (price only, no tweets)
    (
        rf_ema_closing_prices_btc,
        rf_ema_daily_return_volatility_btc,
        rf_bollinger_closing_prices_btc,
        rf_predicted_vs_actual_btc,
        rf_predicted_vs_actual_last_ten_btc,
        rf_cumulative_return_btc,
        rf_ema_closing_prices_doge,
        rf_ema_daily_return_volatility_doge,
        rf_bollinger_closing_prices_doge,
        rf_predicted_vs_actual_doge,
        rf_predicted_vs_actual_last_ten_doge,
        rf_cumulative_return_doge,
    ) = make_random_forest()

    # Plots for Fixed Strategy
    (
        entry_exit_price_plot_btc,
        entry_exit_portfolio_plot_btc,
        portfolio_evaluation_table_btc,
        entry_exit_price_plot_doge,
        entry_exit_portfolio_plot_doge,
        portfolio_evaluation_table_doge
    ) = algo_trading_fixed_strategy()

    # Plots for RNN strategy
    (
        rnn_predicted_positive_return_curve_btc, 
        rnn_cumulative_return_plot_btc,
        rnn_predicted_positive_return_curve_doge, 
        rnn_cumulative_return_plot_doge
    ) = load_algo_trading_result_rnn()

    # Plots for Random Forest strategy
    (
        rf_predicted_positive_return_curve_btc, 
        rf_cumulative_return_plot_btc,
        rf_predicted_positive_return_curve_doge, 
        rf_cumulative_return_plot_doge
    ) = load_algo_trading_result_rf()

    # Plots for Google Trends BTC
    (
        google_predicted_positive_return_curve_btc, 
        google_cumulative_return_plot_btc
    ) = load_trading_result_google_search()
    
    
    ## Create a dashboard (lower half interface in the app) to visualise the plots
    # Return the dashboard appropriate for the ticker selected by the user
    if Source == 'Elon Musk Tweets' and Ticker == 'Bitcoin':
        dashboard = build_dashboard_btc(
                                            tweeting_price_curve_btc, 
                                            cumulative_return_curve_btc,

                                            rf_ema_closing_prices_btc,
                                            rf_ema_daily_return_volatility_btc,
                                            rf_bollinger_closing_prices_btc,
                                            rf_predicted_vs_actual_btc,
                                            rf_predicted_vs_actual_last_ten_btc,
                                            rf_cumulative_return_btc,

                                            entry_exit_price_plot_btc,
                                            entry_exit_portfolio_plot_btc,
                                            portfolio_evaluation_table_btc,

                                            rnn_predicted_positive_return_curve_btc, 
                                            rnn_cumulative_return_plot_btc,

                                            rf_predicted_positive_return_curve_btc, 
                                            rf_cumulative_return_plot_btc
                                        )

    
    if Source == 'Elon Musk Tweets' and Ticker == 'Dogecoin':
        dashboard = build_dashboard_doge(
                                            tweeting_price_curve_doge,
                                            cumulative_return_curve_doge,

                                            rf_ema_closing_prices_doge,
                                            rf_ema_daily_return_volatility_doge,
                                            rf_bollinger_closing_prices_doge,
                                            rf_predicted_vs_actual_doge,
                                            rf_predicted_vs_actual_last_ten_doge,
                                            rf_cumulative_return_doge,

                                            entry_exit_price_plot_doge,
                                            entry_exit_portfolio_plot_doge,
                                            portfolio_evaluation_table_doge,

                                            rnn_predicted_positive_return_curve_doge, 
                                            rnn_cumulative_return_plot_doge,

                                            rf_predicted_positive_return_curve_doge, 
                                            rf_cumulative_return_plot_doge
                                        )
    
    if Source == 'Google Trends Data' and Ticker == 'Bitcoin':
        dashboard = build_dashboard_google_trends_btc(
                                                        price_curve_btc,
                                                        cumulative_return_curve_btc, 
                                                        
                                                        google_predicted_positive_return_curve_btc, 
                                                        google_cumulative_return_plot_btc
                                                    )
    
    
    # Return the dashboard
    return dashboard


# Make interactive drop-down lists which users can choose a social source and crypto
user_input = interact_manual(
                                analyse, 
                                Source=['Select Source', 'Elon Musk Tweets', 'Google Trends Data'], 
                                Ticker=['Select Ticker', 'Bitcoin', 'Dogecoin']
                            )

# Make the upper half interface in the app
interface = pn.Column(
                        '# SillyCon App',
                        '## Analyse correlations between Social Sources and Crypto Prices',
                        '### Select a Source and Crypto Ticker, then press Run Interact', 
                        user_input
                    )

# Launch the app in browser
interface.show()