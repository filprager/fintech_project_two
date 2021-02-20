# Import libraries
from panel.interact import interact_manual
import panel as pn

# Import self-made functiions
from fetch_data import function1
from process_data import function2
from visualise_data import function3

# Analyse the correlation between the chosen entrepreneur and stock/crypto
def analyse(entrepreneur, ticker):
    # Fetch the data
    df_raw = function1(entrepreneur, ticker)

    # Process the data
    df_processed = function2(df_raw)

    # Create a dashboard to visualise the data
    dashboard = function3(df_processed, entrepreneur, ticker)

    # Return the dashboard
    return dashboard

# Make interactive drop-down lists which users can choose an entrepreneur and stock/crypto from
user_input = interact_manual(
                                analyse, 
                                entrepreneur=['Elon Musk', 'Jack Dorsey', 'Mark Cuban', 'DeepFuckingValue'], 
                                ticker=['Bitcoin', 'Dogecoin', 'GME']
                            )

# Make the interface for the app
interface = pn.Column(
                        '## Analyse the correlation between your chosen entrepreneur and stock/crypto',
                        '### Feel free to add whatever you want to display here : )', 
                        user_input
                    )

# Launch the app in the browser
interface.show()