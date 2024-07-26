

import dash
from dash import Dash, dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import plotly.express as px
from utils.cache import get_data
from utils.config import BLUE, PINK, FORESTGREEN, GRAPHCONFIG

NAME = "Multivariate Regression"
PATH = "/multivariate-regression"

layout = html.Div()

x_dropdown = dcc.Dropdown(
    id="x-axis",
    options=[
        {'label': 'Age', 'value': 'age'},
        {'label': 'Years of Experience', 'value': "YoE"}
    ],
    value="age",
    clearable=False,
)

color_dropdown = dcc.Dropdown(
    id="color-by",
    options=[
        {'label': 'Gender', 'value': 'gender'},
        {'label': 'Race', 'value': 'race'},
        {'label': 'Education', 'value': 'education'},
        {'label': 'Level', 'value': 'level'},
    ],
    value="gender",
    clearable=False,
)

inputs = html.Div(id="inputs", style={'width': '40%', 'display': 'inline-block'}, children=[
    x_dropdown,
    color_dropdown,
    html.Button('Toggle Trendline', id='toggle', n_clicks=0),
])


def page_layout():
    layout = html.Div(id="layout", children=[
        dbc.Container([
            inputs,
            dcc.Graph(id='scatterplot', config=GRAPHCONFIG)
        ], className="my-4"),
    ])

    return layout

@callback(
    Output("scatterplot", 'figure'),
    Input('x-axis', 'value'),
    Input('color-by', 'value'),
    Input('toggle', 'n_clicks'),
    State("session", "data")
)

def update_scatter(x_axis, color_by, n_clicks, data):
    session_id = data.get('session_id', None)

    print("get data for update scatter")
    df = get_data(session_id)
    if df is None:
        print("was none!")
        return {}
    print("was NOT none!")

    if color_by == 'gender':
        color_discrete_map = {'Male': BLUE, 'Female': PINK, 'Other': FORESTGREEN}
    else:
        color_discrete_map = None

    if n_clicks % 2 == 1:
        trendline = "ols"
    else:
        trendline = None

    fig = px.scatter(df, x=x_axis, y='pay', color=color_by, color_discrete_map=color_discrete_map,
                     labels={'pay': "Pay", x_axis: x_axis.capitalize(), color_by: color_by.capitalize()},
                     template="simple_white", trendline=trendline)

    fig.update_traces(marker_opacity=0.3)

    fig.update_layout(title=f"Pay vs. {x_axis.capitalize()} by {color_by.capitalize()}",
                      xaxis_title=x_axis.capitalize(), yaxis_title="Pay")
    return fig

dash.register_page(
    __name__,
    path=PATH,
    title=NAME,
    name=NAME,
    order=4,
    layout=layout,
    default=page_layout(),
)
