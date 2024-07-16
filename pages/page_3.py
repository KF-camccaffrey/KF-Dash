

import dash

from dash import Dash, dcc, html, Input, Output, State, callback
import os
import plotly.express as px
from scipy.stats import kruskal
from scikit_posthocs import posthoc_dunn
import pandas as pd
from utils.config import GRAPHCONFIG
from utils.cache import get_data

NAME = "Statistical Tests"
PATH = "/statistical-tests"

dash.register_page(
    __name__,
    path=PATH,
    title=NAME,
    name=NAME,
    order=3
)

heatmap = dcc.Graph(figure={}, id='heatmap', config=GRAPHCONFIG)


layout = html.Div([
    dcc.Store(id='session', storage_type='session'),
    heatmap,
])


@callback(
        Output("heatmap", "figure"),
        #Input("url", "pathname"),
        Input('session', 'data')
)
def update_heatmap(data):
    session_id = data.get('session_id', None)
    df = get_data(session_id)

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
