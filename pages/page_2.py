

import dash
from dash import Dash, dcc, html, Input, Output, State, callback, dash_table
from utils import generator
from utils.config import BLUE, PINK, GRAPHCONFIG, HOVERLABEL
import os

import plotly.express as px
import plotly.figure_factory as ff
import numpy as np
import pandas as pd
import dash_bootstrap_components as dbc
from utils.cache import get_data

NAME = "Visualizations"
PATH = "/visualizations"

dash.register_page(
    __name__,
    path=PATH,
    title=NAME,
    name=NAME,
    order=2
)

data_graph = dcc.Graph(figure={}, id='generate-chart', config=GRAPHCONFIG)
data_table = dash_table.DataTable(id='data-table', data=[], page_size=10)


education_levels = ['Secondary', 'Bachelor', 'Master', 'Doctorate']
level_levels = ['Entry', 'Low', 'Mid', 'Senior']
race_levels = ['Male', 'Female', 'Other']


boxplots = html.Div([
    html.H1("Boxplot with Dropdown"),
    dcc.Dropdown(
        id='category-dropdown',
        options=[
            {'label': 'Education Level', 'value': 'education'},
            {'label': 'Job Level', 'value': 'level'},
            {'label': 'Race/Ethnicity', 'value': 'race'},
        ],
        value='education',  # Default value
        style={'width': '50%'}
    ),
    dcc.Graph(
        figure={},
        id='boxplot',
        #style={'height': '800px'}
    )
])


layout = html.Div([
    dcc.Store(id='session', storage_type='session'),

    dbc.Row([
        dbc.Col(data_graph, width=6),
        dbc.Col(data_table, width=6),
    ]),
    boxplots

])


@callback(
        Output("generate-chart", "figure"),
        #Input("url", "pathname"),
        Input('session', 'data')
)
def update_chart(data):
    session_id = data.get('session_id', None)

    print(f"update chart session iD: {session_id}")

    print("get_data in update_chart()")
    df = get_data(session_id)

    if df is None:
        return {}

    fig = px.histogram(df, x='pay', color='gender', labels={'pay': 'Pay', 'gender': 'Gender'},
                       barmode="overlay", hover_data={"gender": False},
                       color_discrete_map={'Male': BLUE, 'Female': PINK}, opacity=0.4,
                       template="simple_white")

    fig.update_traces(hoverlabel=HOVERLABEL, hovertemplate="<b>%{y}</b>")
    fig.update_layout(hovermode="x")

    return fig




@callback(
    Output("data-table", "data"),
    #Input("generate-chart", "figure"),
    #Input("url", "pathname"),
    Input('session', 'data'),
)
def update_table(data):
    session_id = data.get('session_id', None)
    print(f"update table session iD: {session_id}")

    print("get_data in update_table()")
    df = get_data(session_id)
    if df is None:
        return []
    else:
        return df.to_dict("records")




# Define callback to update the boxplot based on dropdown selection
@callback(
    Output('boxplot', 'figure'),
    Input('category-dropdown', 'value'),
    State('session', 'data')
)
def update_boxplot(selected_category, data):
    # Create the boxplot using Plotly
    print(selected_category)
    print("get_data in update_boxplot()")
    session_id = data.get('session_id', None)
    df = get_data(session_id)

    if df is None:
        return {}
    else:
        print(df)

    fig = px.box(df, x=selected_category, y='pay', color='gender',
                 category_orders={selected_category: sorted(df[selected_category].unique())},
                 title=f'Distribution of Pay by {selected_category} and Gender',
                 labels={'Pay': 'Pay ($)', selected_category: selected_category, 'Gender': 'Gender'},
                 color_discrete_map={'Male': BLUE, 'Female': PINK, "Other": "green"},
                 points=False,  # Show all points
                 template="simple_white"
    )

    # Update layout for better visualization
    fig.update_layout(
        boxmode='group',  # Group boxes together
        boxgap=0.3,  # Reduce the gap between boxes
        boxgroupgap=0.2,  # Reduce the gap between different groups of boxes
        yaxis=dict(
            title='Pay ($)'
        ),
        xaxis=dict(
            title=f'{selected_category}'
        )
    )

    return fig
