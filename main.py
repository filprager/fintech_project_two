# Import libraries
from panel.interact import interact_manual
import panel as pn

# Import self-made functiions
from process_data import process_data
from build_dashboard import build_dashboard

# Analyse the correlation between the chosen entrepreneur and stock/crypto
def analyse(entrepreneur, ticker):
    # Process the data
    plot1, plot2, plot3, table1, table2, table3 = process_data()

    # Create a dashboard to visualise the data
    dashboard = build_dashboard(plot1, plot2, plot3, table1, table2, table3, entrepreneur, ticker)

    # Return the dashboard
    return dashboard

# Make interactive drop-down lists which users can choose an entrepreneur and stock/crypto from
user_input = interact_manual(
                                analyse, 
                                entrepreneur=['Elon Musk'], 
                                ticker=['Bitcoin', 'Dogecoin']
                            )

# Make the interface for the app
interface = pn.Column(
                        '## Analyse the correlation between your chosen entrepreneur and stock/crypto',
                        '### Feel free to add whatever you want to display here : )', 
                        user_input
                    )

# Launch the app in browser
interface.show()