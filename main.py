# Import libraries
from panel.interact import interact_manual
import panel as pn

# Import self-made functiions
from fetch_data import function1
from process_data import function2
from visualise_data import function3

# Analyse the correlation between the chosen entrepreneur and crypto
def analyse(entrepreneur, crypto):
    # Fetch the data
    df_raw = function1(entrepreneur, crypto)

    # Process the data
    df_processed = function2(df_raw)

    # Create a dashboard to visualise the data
    dashboard = function3(df_processed, entrepreneur, crypto)

    # Return the dashboard
    return dashboard

# Make interactive drop-down lists which users can choose entrepreneur and crypto from
user_input = interact_manual(
                                analyse, 
                                entrepreneur=['Elon Musk', 'Jack Dorsey'], 
                                crypto=['Bitcoin', 'Dogecoin']
                            )

# Make the interface for the app
interface = pn.Column(
                        '## Analyse the correlation between your chosen entrepreneur and crypto',
                        '### Feel free to add whatever you want to display here : )', 
                        user_input
                    )

# Launch the app in the browser
interface.show()