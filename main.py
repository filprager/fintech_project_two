# Import libraries
from panel.interact import interact_manual
import panel as pn

# Import self-made functiions
from build_dashboard import build_dashboard
from process_data import make_tweeting_price_curve, make_cumulative_curve, make_price_curve, function_random_forest

# Analyse the correlation between the chosen entrepreneur and stock/crypto
def analyse(entrepreneur, ticker):
    # Process the data
    tweeting_price_curve_doge, tweeting_price_curve_btc = make_tweeting_price_curve()
    cumulative_return_curve_doge,cumulative_return_curve_btc = make_cumulative_curve()
    price_curve_btc, price_curve_doge = make_price_curve()
    # plot1, plot2, plot3, plot4, plot5, plot6, plot7, table1, table2, table3 = function_random_forest()

    # Create a dashboard to visualise the data
    dashboard = build_dashboard(
                                tweeting_price_curve_doge, 
                                tweeting_price_curve_btc, 
                                cumulative_return_curve_doge, 
                                cumulative_return_curve_btc, 
                                price_curve_btc, 
                                price_curve_doge, 
                                entrepreneur, 
                                ticker
                                )

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