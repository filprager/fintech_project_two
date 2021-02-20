import panel as pn

def function3(df_processed, entrepreneur, crypto):

    dashboard = pn.Column(
        f'You have chosen to analyse the correlation between {entrepreneur} and {crypto} prices', 
        'I am your wanted result : )'
        )
    
    return dashboard