

import dash
from dash import Dash, dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np

NAME = "Remediation Modeling"
PATH = "/remediation-modeling"

layout = html.Div()

font_test = html.Div([
    html.P("The fat cat jumped over the white fence.", style={"font-family": "Gotham Thin"}),
    html.P("The fat cat jumped over the white fence.", style={"font-family": "Gotham ExtraLight"}),
    html.P("The fat cat jumped over the white fence.", style={"font-family": "Gotham Light"}),
    html.P("The fat cat jumped over the white fence.", style={"font-family": "Gotham"}),
    html.P("The fat cat jumped over the white fence.", style={"font-family": "Gotham Medium"}),
    html.P("The fat cat jumped over the white fence.", style={"font-family": "Gotham Bold"}),
    html.P("The fat cat jumped over the white fence.", style={"font-family": "Gotham Black"}),
    html.P("The fat cat jumped over the white fence.", style={"font-family": "Gotham Ultra"}),
])

def page_layout():
    layout = html.Div(id="layout", children=[
        dbc.Container([
            html.P("This is page 5"),
        ], className="my-4"),
    ])

    return layout


dash.register_page(
    __name__,
    path=PATH,
    title=NAME,
    name=NAME,
    order=5,
    layout=layout,
    default=page_layout(),
)
