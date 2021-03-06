import panel as pn


def build_dashboard_btc(
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
):

    '''
    Make panel layout for BTC
    Returns: Panel layout
    '''
    # Content for tab_one
    tab_one = pn.Column(
        tweeting_price_curve_btc,
        cumulative_return_curve_btc, 

        # Word cloud
        '### What Elon Musk recently said about Bitcoin on Twitter ![Image](https://github.com/filprager/fintech_project_two/blob/main/image/wordcloud_bitcoin.png?raw=true)'
    )
    
    # Content for tab_two
    tab_two = pn.Column(
        rf_ema_closing_prices_btc,
        rf_ema_daily_return_volatility_btc,
        rf_bollinger_closing_prices_btc,
        rf_predicted_vs_actual_btc,
        rf_predicted_vs_actual_last_ten_btc,
        rf_cumulative_return_btc,
    )
    
    # Content for tab_three
    tab_three = pn.Column(
        '## Buy whenever Elon Musk tweets about Bitcoin, and sell after 24 hours',
        entry_exit_price_plot_btc,
        entry_exit_portfolio_plot_btc,
        portfolio_evaluation_table_btc,
    )

    # Content for tab_four
    tab_four = pn.Column(
        '## Long when the model predicts the price to rise, and short when the model predicts the price to fall',
        rnn_predicted_positive_return_curve_btc, 
        rnn_cumulative_return_plot_btc,
    )
    
  
    # Content for tab_five
    tab_five = pn.Column(
        '## Long when the model predicts the price to rise, and short when the model predicts the price to fall',
        rf_predicted_positive_return_curve_btc, 
        rf_cumulative_return_plot_btc,
    )

    # Combined dashboard of all tabs
    dashboard_btc = pn.Tabs(
        ('General Insights', tab_one),
        ('Random Forest - Price Only', tab_two),
        ('Fixed Strategy', tab_three),
        ('RNN Strategy', tab_four),
        ('RF Strategy', tab_five),
    )
    
    return dashboard_btc


    # _________________________________________________________________________________________________________________________________________________________ 


def build_dashboard_doge(
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
):

    '''
    Make panel layout for Doge
    Returns: Panel layout
    '''
    # Content for tab_one
    tab_one = pn.Column(
        tweeting_price_curve_doge,
        cumulative_return_curve_doge,

        # Word cloud
        '### What Elon Musk recently said about Dogecoin on Twitter ![Image](https://github.com/filprager/fintech_project_two/blob/main/image/wordcloud_doge.png?raw=true)'

    )
    
    # Content for tab_two
    tab_two = pn.Column(
        rf_ema_closing_prices_doge,
        rf_ema_daily_return_volatility_doge,
        rf_bollinger_closing_prices_doge,
        rf_predicted_vs_actual_doge,
        rf_predicted_vs_actual_last_ten_doge,
        rf_cumulative_return_doge,
    )
    
    # Content for tab_three
    tab_three = pn.Column(
        '## Buy whenever Elon Musk tweets about Dogecoin, and sell after 24 hours',
        entry_exit_price_plot_doge,
        entry_exit_portfolio_plot_doge,
        portfolio_evaluation_table_doge,
    )

    # Content for tab_four
    tab_four = pn.Column(
        '## Long when the model predicts the price to rise, and short when the model predicts the price to fall',
        rnn_predicted_positive_return_curve_doge, 
        rnn_cumulative_return_plot_doge,
    )
    
  
    # Content for tab_five
    tab_five = pn.Column(
        '## Long when the model predicts the price to rise, and short when the model predicts the price to fall',
        rf_predicted_positive_return_curve_doge, 
        rf_cumulative_return_plot_doge
    )

    # Combined dashboard of all tabs
    dashboard_doge = pn.Tabs(
        ('General Insights', tab_one),
        ('Random Forest - Price Only', tab_two),
        ('Fixed Strategy', tab_three),
        ('RNN Strategy', tab_four),
        ('RF Strategy', tab_five),
    )
    
    return dashboard_doge


    # _________________________________________________________________________________________________________________________________________________________ 


def build_dashboard_google_trends_btc(
                                        price_curve_btc,
                                        cumulative_return_curve_btc, 

                                        google_predicted_positive_return_curve_btc, 
                                        google_cumulative_return_plot_btc
):

    '''
    Make panel layout for Google Trends BTC
    Returns: Panel layout
    '''
    # Content for tab_one
    tab_one = pn.Column(
        price_curve_btc,
        cumulative_return_curve_btc
    )
    

    # Content for tab_two
    tab_two = pn.Column(
        google_predicted_positive_return_curve_btc,
        google_cumulative_return_plot_btc
    )
    
    # Combined dashboard of all tabs
    dashboard_google_trends_btc = pn.Tabs(
        ('General Insights', tab_one),
        ('RNN Strategy', tab_two),
    )
    
    return dashboard_google_trends_btc