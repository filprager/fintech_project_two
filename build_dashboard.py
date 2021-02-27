import panel as pn

def function2(plot1, plot2, plot3, table1, table2, table3, entrepreneur, ticker):

    dashboard = pn.Column(
        f'You have chosen to analyse the correlation between {entrepreneur} and {ticker} prices', 
        'I am your wanted result : )',
        'Table A'
        )
    
    return dashboard