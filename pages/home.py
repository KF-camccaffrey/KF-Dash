

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

# Page Properties
NAME = "Home"
PATH = "/"

# Important File Paths
README = "README.md"
BACKGROUND = "/assets/images/favicon.png"

layout = html.Div()

def page_layout():
    with open(README, 'r', encoding='utf-8') as f:
        text = f.read()
    layout = dbc.Container(
        className="h-100",
        style={
            "display": "flex",
            "justifyContent": "center",
            "alignItems": "left",
            #"backgroundImage": f"url('{BACKGROUND}')",
            #"backgroundSize": "cover",
            #"padding": "20px",
        },
        children=[
            dbc.Row(dbc.Col(
                dcc.Markdown(text),
                width=12,
                className="markdown-container",
                style={
                    #"font-family": "Geist, sans-serif",
                    "backgroundColor": "rgba(255, 255, 255, 0.0)",
                    "padding": "10px",
                    "borderRadius": "10px",
                }
            ))
        ]
    )
    return layout
    """
    layout = dbc.Container(
        dbc.Row(
            dbc.Col(
                html.Div([
                    html.H1("Welcome to Your App"),
                    html.P("This app is designed to ... (explain the purpose of your app)"),
                    html.H2("Instructions:"),
                    html.P("1. Instruction 1"),
                    html.P("2. Instruction 2"),
                    html.P("3. Instruction 3"),
                    html.P("Feel free to explore and use the app!"),
                ], style={"text-align": "center", "margin-top": "20%"}),

                width={"size": 6, "offset": 3},  # Center the column
            ),
        ),
        className="h-100",  # Make sure the container takes full height
    )
    """




dash.register_page(
    __name__,
    path=PATH,
    title=NAME,
    name=NAME,
    order=0,
    layout=layout,
    default=page_layout(),
)
