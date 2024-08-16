
"""
File Name: page_5.py
Author: Connor McCaffrey
Date: 8/16/2024

Description:
    - This file contains the layout and callback functions used for the "Interactions" page.
"""

import dash
from dash import dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc
import plotly.express as px
from utils.cache import query_data, query_comparisons
from utils.config import BLUE, PINK, FORESTGREEN, GRAPHCONFIG, EMPTYFIG

NAME = "Interactions"
PATH = "/interactions"

layout = html.Div()

x_dropdown = dcc.Dropdown(
    id="x-axis",
    clearable=False,
)

color_dropdown = dcc.Dropdown(
    id="color-by",
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
            dcc.Graph(id='scatterplot', config=GRAPHCONFIG, figure=EMPTYFIG)
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
    if x_axis == "none" or color_by == "none":
        return EMPTYFIG

    session_id = data.get('session_id', None)
    df, _ = query_data(session_id)
    if df is None:
        return EMPTYFIG

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



@callback(
    Output("x-axis", "value"),
    Output("x-axis", "options"),
    Input("session", "data"),
    State("x-axis", "value"),
)
def set_x(data, x):
    session_id = data.get("session_id", None)
    comparisons = query_comparisons(session_id)

    if comparisons is None:
        return "none", [{"label": "No Data Available", "value": "none"}]

    quant = comparisons["quantitative"]
    options = [{"label": f"{q.capitalize()}", "value": q} for q in quant]

    if not options:
        return "none", [{"label": " No Data Available", "value": "none"}]

    if x:
        return x, options
    else:
        return options[0]["value"], options


@callback(
    Output("color-by", "value"),
    Output("color-by", "options"),
    Input("session", "data"),
    State("color-by", "value"),
)

def set_color(data, category):
    session_id = data.get("session_id", None)
    comparisons = query_comparisons(session_id)

    if comparisons is None:
        return "none", [{"label": "No Data Available", "value": "none"}]

    categorical = comparisons["categorical"]
    keys = list(categorical.keys())
    options = [{"label": f"{k.capitalize()}", "value": k} for k in keys]

    if not options:
        return "none", [{"label": " No Data Available", "value": "none"}]

    if category:
        return category, options
    else:
        return options[0]["value"], options



dash.register_page(
    __name__,
    path=PATH,
    title=NAME,
    name=NAME,
    order=5,
    layout=layout,
    default=page_layout(),
)
