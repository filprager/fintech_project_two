import panel as pn

def function3(df_processed, entrepreneur, ticker):

    dashboard = pn.Column(
        f'You have chosen to analyse the correlation between {entrepreneur} and {ticker} prices', 
        'I am your wanted result : )'
        )
    
    return dashboard