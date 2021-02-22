# Import libraries
from panel.interact import interact_manual
import panel as pn

# Import self-made functiions
from fetch_data import function1
from process_data import function2
from build_dashboard import function3

# Analyse the correlation between the chosen entrepreneur and stock/crypto
def analyse(entrepreneur, ticker):
    # Fetch the data
    df = function1(entrepreneur, ticker)

    # Process the data
    plot1, plot2, plot3, table1, table2, table3 = function2(df)

    # Create a dashboard to visualise the data
    dashboard = function3(plot1, plot2, plot3, table1, table2, table3, entrepreneur, ticker)

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

# Launch the app in browser
interface.show()