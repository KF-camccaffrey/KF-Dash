
"""
File Name: home.py
Author: Connor McCaffrey
Date: 8/16/2024

Description:
    - This file contains the layout and callback functions used for the "Home" page.
"""

import dash
from dash import dcc, html
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
        },
        children=[
            dbc.Row(dbc.Col(
                dcc.Markdown(text),
                width=12,
                className="markdown-container",
                style={
                    "backgroundColor": "rgba(255, 255, 255, 0.0)",
                    "padding": "10px",
                    "borderRadius": "10px",
                }
            ))
        ]
    )
    return layout

dash.register_page(
    __name__,
    path=PATH,
    title=NAME,
    name=NAME,
    order=0,
    layout=layout,
    default=page_layout(),
)
