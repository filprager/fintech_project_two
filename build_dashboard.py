import panel as pn

def build_dashboard(plot1, plot2, plot3, plot4, plot5, plot6, plot7, plot8, plot9, plot10, plot11, plot12, plot13):

    
    # Content for tab_one
    tab_one = pn.Column(
        plot1,
        plot2,
        plot3
    )
    
    # Content for tab_two
    tab_two = pn.Column(
        plot4,
        plot5, 
        plot6
    )
    
    # Content for tab_three
    tab_three = pn.Column(
        plot7,
        plot8,
        plot9
    )

    # Content for tab_four
    tab_four = pn.Column(
        plot10,
        plot11,
        plot12,
        plot13
    )
    
    
    # Combined dashboard of all tabs
    dashboard = pn.Tabs(
        ('Tab 1 name here', tab_one),
        ('Tab 2 name here', tab_two),
        ('Tab 3 name here', tab_three),
        ('Tab 4 name here', tab_four)
    )
    
    return dashboard