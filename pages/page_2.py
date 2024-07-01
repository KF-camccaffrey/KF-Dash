

import dash
from dash import Dash, dcc, html, Input, Output, State
from utils import generator
from utils.config import BLUE, PINK, GRAPHCONFIG, HOVERLABEL


dash.register_page(__name__)

from dash import Dash, dcc, html, Input, Output, callback
import plotly.express as px
import plotly.figure_factory as ff
import numpy as np
import pandas as pd

# Incorporate data
df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/gapminder2007.csv')


layout = html.Div([
    dcc.Store(id='session', storage_type='session'),
    html.P("test", id="example-output2", style={"verticalAlign": "middle"}),

    dcc.Graph(figure={}, id='generate-chart', config=GRAPHCONFIG),
])




@callback(
        #Output("example-output2", "children"),
        Output("generate-chart", "figure"),
        Input("url", "pathname"),
        State('session', 'data')
)
def update_generate_chart(pathname, data):
    if pathname == '/page-2':
        data = data or {}

        male_pay_range = data.get('male_pay_range', [60000, 80000])
        female_pay_range = data.get('female_pay_range', [60000, 80000])
        gender_ratio = data.get("gender_ratio", 50)

        print(f"Male: {male_pay_range}, Female: {female_pay_range}, p_M = {gender_ratio}%")
        df = generator.basic(male_pay_range, female_pay_range, gender_ratio)
        print(df)

        fig = px.histogram(df, x='pay', color='sex', labels={'pay': 'Pay', 'sex': 'Sex'},
                           barmode="overlay", hover_data={"sex": False},
                           color_discrete_map={'Male': BLUE, 'Female': PINK}, opacity=0.7,
                           template="simple_white")

        fig.update_traces(hoverlabel=HOVERLABEL, hovertemplate="<b>%{y}</b>")
        fig.update_layout(hovermode="x")

        return fig
    else:
        return {}
