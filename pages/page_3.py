

import dash
from dash import Dash, dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np

import plotly.express as px
from scipy.stats import kruskal
from scikit_posthocs import posthoc_dunn
from utils.config import GRAPHCONFIG, RED, GRAY, YELLOW, BASICCOMPS, EMPTYFIG
from utils.cache import query_data, query_comparisons
from utils.comparisons import create_comparisons, effect_bars


NAME = "Statistical Tests"
PATH = "/statistical-tests"

heatmap = dcc.Graph(figure={}, id='heatmap', config=GRAPHCONFIG)

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
        options=[
            {'label': "Education", 'value': 'education'},
            {'label': "Gender", 'value': 'gender'},
            {'label': "Race/Ethnicity", 'value': 'race'},
            {'label': "Job Level", 'value': 'level'},
            {'label': "Department", 'value': 'department'},
            {'label': "All", 'value': 'all'},
        ],
        value='gender',
        clearable=False,
        #options=[
        #    {'label': 'Difference in Medians ($)', 'value': 'median_diff'},
        #    {'label': 'Difference in Medians (%)', 'value': 'median_perc'},
        #],
        #value='median_diff',  # Default value
        style={'width': '50%'}
    ),
    dcc.Graph(
        figure={},
        id='barchart',
        #style={'height': '800px'}
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


"""
@callback(
        Output("heatmap", "figure"),
        #Input("url", "pathname"),
        Input('session', 'data')
)
def update_heatmap(data):
    session_id = data.get('session_id', None)
    df = query_data(session_id)

    male_pay = df[df['gender'] == 'Male']['pay']
    female_pay = df[df['gender'] == 'Female']['pay']
    other_pay = df[df['gender'] == 'Other']['pay']

    # Conduct Kruskal-Wallis test
    stat, p_value = kruskal(male_pay, female_pay, other_pay)
    print(f'Kruskal-Wallis Statistic: {stat}, P-value: {p_value}')

    # Conduct Dunn's test
    posthoc_results = posthoc_dunn(df, val_col='pay', group_col='gender', p_adjust='holm')

    fig = px.imshow(posthoc_results)

    return fig
"""


# Define callback to update the boxplot based on dropdown selection
@callback(
    Output('barchart', 'figure'),
    Input('category-dropdown', 'value'),
    State('session', 'data')
)
def update_barchart(selected_category, data):
    # Create the boxplot using Plotly
    #print(selected_category)
    #print("query_data in update_barchart()")
    session_id = data.get('session_id', None)
    timestamp = data.get('timestamp', None)

    comparisons = query_comparisons(session_id, timestamp, BASICCOMPS)

    if comparisons is None:
        return {}

    fig = effect_bars(comparisons, category=selected_category, method='median_comp')
    return fig


"""
@callback(
    Output('layout', 'children'),
    Input('url', 'pathname'),
    State('session', 'data'),
)
def validate(path, data):
    if path == "/statistical-tests":
        valid = data.get("valid", False)
        if not valid:
            return no_data_layout()
        else:
            return page_layout().children
    else:
        raise dash.exceptions.PreventUpdate
"""

dash.register_page(
    __name__,
    path=PATH,
    title=NAME,
    name=NAME,
    order=3,
    layout=layout,
    default=page_layout(),
)
