import dash

from dash import Dash, dcc, html, Input, Output, State, callback
import os
import plotly.express as px
from scipy.stats import kruskal
from scikit_posthocs import posthoc_dunn
import pandas as pd
from utils.config import GRAPHCONFIG, RED, GRAY, YELLOW
from utils.cache import get_data
from test import effect_sizes, effect_bars
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
