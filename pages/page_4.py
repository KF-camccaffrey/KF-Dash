
"""
File Name: page_4.py
Author: Connor McCaffrey
Date: 8/16/2024

Description:
    - This file contains the layout and callback functions used for the "Pairwise Comparisons" page.
"""

import dash
from dash import dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc
import numpy as np
from utils.config import GRAPHCONFIG, RED, GRAY, EMPTYFIG
from utils.cache import query_comparisons
from utils.comparisons import effect_bars


NAME = "Pairwise Comparisons"
PATH = "/pairwise-comparisons"


barchart = html.Div([
    html.Div([
        html.H3("Pairwise Comparisons"),
        html.Span("Statistically significant differences will appear in "),
        html.Span("red", style={'color': RED, 'font-weight': 'bold'}),
        html.Span(", and insignificant differences in "),
        html.Span("gray", style={'color': GRAY, 'font-weight': 'bold'}), html.Span("."),
    ], className="my-5"),

    dbc.Label("Select Variable", html_for="category-dropdown", className="input-group-label", style={'font-weight': 'bold'}),
    dcc.Dropdown(
        id='category-dropdown',
        clearable=False,
        style={'width': '50%'}
    ),
    dcc.Graph(
        figure=EMPTYFIG,
        id='barchart',
    )
])

layout = html.Div()

def page_layout():
    layout = html.Div(id="layout", children=[
        dbc.Container([
            barchart,
        ], className="my-4"),
    ])

    return layout


@callback(
    Output('barchart', 'figure'),
    Input('category-dropdown', 'value'),
    State('session', 'data')
)
def update_barchart(selected_category, data):
    session_id = data.get('session_id', None)
    comparisons = query_comparisons(session_id)

    if comparisons is None:
        return {}

    fig = effect_bars(comparisons, category=selected_category, method='median_comp')
    return fig




@callback(
    Output("category-dropdown", "value"),
    Output("category-dropdown", "options"),
    Input("session", "data"),
    State("category-dropdown", "value"),
)

def set_category(data, category):
    session_id = data.get("session_id", None)
    comparisons = query_comparisons(session_id)

    if comparisons is None:
        return "none", [{"label": "No Data Available", "value": "none"}]

    categorical = comparisons["categorical"]
    keys = list(categorical.keys())
    options = [{"label": f"{k.capitalize()}", "value": k} for k in keys if not all(np.isnan(categorical[k]["median_comp"][1:]))]

    if not options:
        return "none", [{"label": " No Data Available", "value": "none"}]
    elif len(options) > 1:
        options = options + [{'label': "All", 'value': 'all'}]

    if category:
        return category, options
    else:
        return options[0]["value"], options



dash.register_page(
    __name__,
    path=PATH,
    title=NAME,
    name=NAME,
    order=4,
    layout=layout,
    default=page_layout(),
)
