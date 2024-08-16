
"""
File Name: not_found_404.py
Author: Connor McCaffrey
Date: 8/16/2024

Description:
    - This file contains the layout and callback functions used for the "404 Page Not Found" page.
"""
import dash

from dash import html
import dash_bootstrap_components as dbc

layout = html.Div()

def page_layout():
    layout = dbc.Container(
        dbc.Row(
            dbc.Col(
                html.Div([
                    html.H2("404 - Page Not Found"),
                    html.P("The page you are looking for does not exist."),
                    dbc.Button("Go to Homepage", color="primary", href="/"),
                ], style={"text-align": "center", "margin-top": "20%"}),

                width={"size": 6, "offset": 3},  # Center the column
            ),
        ),
        className="h-100",  # Make sure the container takes full height
    )

    return layout


dash.register_page(
    __name__,
    order=-1,
    layout=layout,
    default=page_layout(),
)
