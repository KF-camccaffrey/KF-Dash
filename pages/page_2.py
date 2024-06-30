

import dash
from dash import Dash, dcc, html, Input, Output, State
from utils import generator


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

    dcc.Graph(figure={}, id='generate-chart'),
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
        gender_ratio = data.get("gender_ratio", 0.5)

        m_pay, f_pay = generator.basic(male_pay_range, female_pay_range, gender_ratio)
        fig = ff.create_distplot([m_pay, f_pay], ["Male", "Female"], bin_size=1000)
        #px.histogram(
        #    df, x="pay", color="sex", marginal="box", nbins=30
        #)
        return fig
        #return f"{gender_ratio}, {male_pay_range}, {female_pay_range}"
    else:
        return {}

        #return f"idk {pathname}"


#    mask = df["day"] == day
#    fig = px.bar(df[mask], x="sex", y="total_bill", color="smoker", barmode="group")
