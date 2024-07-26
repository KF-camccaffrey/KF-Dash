

import dash
from dash import Dash, dcc, html, Input, Output, State, callback, dash_table, callback_context
from utils import generator
from utils.config import BLUE, PINK, GRAPHCONFIG, HOVERLABEL
import os

import plotly.express as px
import plotly.figure_factory as ff
import numpy as np
import pandas as pd
import dash_bootstrap_components as dbc
from utils.cache import get_data

from utils.comparisons import create_comparisons, dumbbell_chart

NAME = "Visualizations"
PATH = "/visualizations"

data_graph = dcc.Graph(figure={}, id='generate-chart', config=GRAPHCONFIG)
data_table = dash_table.DataTable(id='data-table', data=[], page_size=10, style_cell={"font-family": "Gotham Thin"},)


education_levels = ['Secondary', 'Bachelor', 'Master', 'Doctorate']
level_levels = ['Entry', 'Low', 'Mid', 'Senior']
race_levels = ['Male', 'Female', 'Other']

"""
boxplots = html.Div([
    html.H3("Grouped Boxplots"),
    #dbc.Label("Select Variable", html_for="category-dropdown", className="input-group-label", style={'font-weight': 'bold'}),
    #dcc.Dropdown(
    #    id='category-dropdown',
    #    options=[
    #        {'label': 'Education Level', 'value': 'education'},
    #        {'label': 'Job Level', 'value': 'level'},
    #        {'label': 'Race/Ethnicity', 'value': 'race'},
    #    ],
    #    value='education',  # Default value
    #    style={'width': '50%'}
    #),
    dcc.Graph(
        figure={},
        id='boxplot',
        #style={'height': '800px'}
    )
])
"""




layout = html.Div()

# Define custom CSS styles
"""
SIDEBAR_STYLE = {
    #"position": "relative",
    "height": "100%",
    #"top": "70px",
    #"left": 0,
    #"bottom": "63px",
    #"padding": "2rem 1rem",
    "background-color": "#f8f9fa",  # Light gray background
}
"""




sideinputs = html.Div(className="pl-5", children=[
    html.Div(className="py-3", children=[
        dbc.Label(html.H5("Details By"), width="auto"),
        dcc.RadioItems(
            id="category-input",
            options=[
                {'label': ' Department', 'value': 'department'},
                {'label': ' Education', 'value': 'education'},
                {'label': ' Level', 'value': 'level'},
                {'label': ' Race', 'value': 'race'},
            ],
            value='department',
        ),
    ]),
    html.Div(className="py-3", children=[
        dbc.Label(html.H5("Sort By"), width="auto"),
        dcc.RadioItems(
            id="sort-input",
            options=[
                {'label': ' A to Z', 'value': 'alpha'},
                {'label': ' Largest Pay Gap', 'value': 'diff'},
                {'label': ' Most Employees', 'value': 'n'},
            ],
            value='alpha',
        ),
    ]),
    html.Div(className="py-3", children=[
        dbc.Label(html.H5("Metric"), width="auto"),
        dcc.RadioItems(
            id='method-input',
            options=[
                {'label': ' Mean', 'value': 'mean'},
                {'label': ' Median', 'value': 'median'},
            ],
            value="mean",
            labelStyle={"display": "inline-block", "padding": "5px"}
        )
    ])
])

dumbbells = dcc.Graph(
    id="dumbbell",
    figure={},
    config=GRAPHCONFIG,
    className="px-3 py-3"
)

boxplots = dcc.Graph(
    figure={},
    id='boxplot',
    config=GRAPHCONFIG,
    className="px-3 py-3"
)

caption = html.Div(id="caption", className="px-3 py-3")
metrics = html.Div(id="metrics", className="px-3 py-3")


def page_layout():
    layout = dbc.Container(id="layout", fluid=True, style={"height": "100%", "padding": "0px"}, children=[
        dbc.Row(style={"height": "100%"}, children=[
            dbc.Col(id="sidebar", width=2, children=[
                sideinputs,
            ]),
            dbc.Col(id="plot-main", width=6, children=[
                html.Div(className="plot-frame", children=[
                    caption,
                    dumbbells,
                ])
            ]),
            dbc.Col(id="plot-side", width=4, children=[
                html.Div(className="plot-frame", children=[
                    metrics,
                    boxplots,
                ])
            ]),
        ])
        #dbc.Container([
        #    dbc.Row([
        #        dbc.Col(data_graph, width=6),
        #        dbc.Col(data_table, width=6),
        #    ]),
        #    boxplots,
        #    dumbbells,
        #]),
        #dumbbells,
        #boxplots
    ])

    return layout


@callback(
        Output("generate-chart", "figure"),
        #Input("url", "pathname"),
        Input('session', 'data')
)
def update_chart(data):
    session_id = data.get('session_id', None)

    #print(f"update chart session iD: {session_id}")

    #print("get_data in update_chart()")
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
    #print(f"update table session iD: {session_id}")

    #print("get_data in update_table()")
    df = get_data(session_id)
    if df is None:
        return []
    else:
        return df.to_dict("records")



"""
# Define callback to update the boxplot based on dropdown selection
@callback(
    Output('boxplot', 'figure'),
    Input('category-dropdown', 'value'),
    State('session', 'data')
)
def update_boxplot(selected_category, data):
    # Create the boxplot using Plotly
    #print(selected_category)
    #print("get_data in update_boxplot()")
    session_id = data.get('session_id', None)
    df = get_data(session_id)

    if df is None:
        return {}


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
"""

@callback(
    Output('boxplot', 'figure'),
    Input('category-input', 'value'),
    Input('dumbbell', 'hoverData'),
    Input('dumbbell', 'clickData'),
    State('session', 'data'),
)
def update_boxplot(category, hoverData, clickData, data):
    session_id = data.get('session_id', None)
    df = get_data(session_id)

    if df is None:
        return {}

    if clickData is None and hoverData is None:
        pass
    elif clickData is None:
        clickData = hoverData
        lvl = clickData["points"][0]['y']
        if lvl == "Overall":
            pass
        elif lvl not in df[category].values:
            pass
        else:
            df =  df[df[category] == lvl]
    else:
        lvl = clickData["points"][0]['y']
        if lvl == "Overall":
            pass
        elif lvl not in df[category].values:
            pass
        else:
            df = df[df[category] == lvl]

    fig = px.box(df, x='gender', y='pay', color='gender',
                 labels={'pay': 'Pay ($)'},
                 color_discrete_map={'Male': BLUE, 'Female': PINK, "Other": "green"},
                 category_orders={"gender": ["Male", "Female", "Other"]},
                 points=False,
                 template="simple_white",
    )

    mean_values = df.groupby('gender')['pay'].mean().reset_index()

    #fig.add_trace(px.scatter(mean_values, x="gender", y="pay",
    #              mode="markers", showlegend=False).data[0])

    # Update layout for better visualization
    fig.update_layout(
        yaxis=dict(
            title='Pay ($)',
            tickfont_family="Gotham",
        ),
        xaxis_title=None,
        showlegend=False,
        margin_t=20,
    )

    return fig

@callback(
    Output('metrics', 'children'),
    Input('category-input', 'value'),
    Input('method-input', 'value'),
    Input('dumbbell', 'hoverData'),
    Input('dumbbell', 'clickData'),
    State('session', 'data'),
)
def update_metrics(category, method, hoverData, clickData, data):
    session_id = data.get('session_id', None)
    df = get_data(session_id)

    if df is None:
        return []

    if clickData is None and hoverData is None:
        lvl = "Overall"
        pass
    elif clickData is None:
        clickData = hoverData
        lvl = clickData["points"][0]['y']
        if lvl == "Overall":
            pass
        elif lvl not in df[category].values:
            lvl = "Overall"
            pass
        else:
            df =  df[df[category] == lvl]
    else:
        lvl = clickData["points"][0]['y']
        if lvl == "Overall":
            pass
        elif lvl not in df[category].values:
            lvl = "Overall"
            pass
        else:
            df = df[df[category] == lvl]

    male = df[df["gender"] == "Male"]
    female = df[df["gender"] == "Female"]
    if method == "mean":
        male_metric = male["pay"].mean()
        female_metric = female['pay'].mean()
    elif method == "median":
        male_metric = male["pay"].median()
        female_metric = female["pay"].median()
    else:
        raise ValueError("Invalid method selected")

    ratio = round(female_metric / male_metric, 2)
    male_metric = round(male_metric, 2)
    female_metric = round(female_metric, 2)
    n_male = len(male)
    n_female = len(female)

    layout = [
        html.H5(f"Level Summary"),
        html.H4(lvl),
        html.Div(id="metric-display", className="mx-2", children=[
            html.H5([
                html.Div("Avg. Men's Pay:", className="metric-label blue"),
                html.Div('${:20,.2f}'.format(male_metric), className="metric-value tabular blue"),
                html.Div("N:", className="n-label blue"),
                html.Div('{:20}'.format(n_male),  className="n-value tabular blue")
            ]),
            html.H5([
                html.Div("Avg. Women's Pay:", className="metric-label pink"),
                html.Div('${:20,.2f}'.format(female_metric), className="metric-value tabular pink"),
                html.Div("N:", className="n-label pink"),
                html.Div('{:20}'.format(n_female),  className="n-value tabular pink")
            ]),
            html.Hr(style={'margin': '10px 0', "width": "350px"}),
            html.H5([
                html.Div("Wage Gap:", className="metric-label"),
                html.Div('${:20,.2f}'.format(abs(male_metric-female_metric)), className="metric-value tabular")
            ]),
        ]),
        html.Div(className="mt-5", style={"textAlign": "center"}, children=[
            "A woman makes ",
            html.Span(f"${ratio}", className="tabular pink"),
            " for every ",
            html.Span("$1.00", className="tabular blue"),
            " a man makes."
        ])
    ]

    return layout

@callback(
    Output('dumbbell', 'clickData'),
    Input('category-input', 'value')
)
def reset_dumbbell_data(category):
    return None

@callback(
    Output('caption', 'children'),
    Input('category-input', 'value'),
    Input('method-input', 'value'),
)
def update_caption(category, method):
    layout = [
        html.H5(f"Difference in {method.capitalize()} Pay Between Men & Women"),
        html.H4(f"By {category.capitalize()}")
    ]
    return layout


@callback(
    Output('dumbbell', 'figure'),
    Input('category-input', 'value'),
    Input('method-input', 'value'),
    Input('sort-input', 'value'),
    State('session', 'data')
)
def update_dumbell(category, method, sort, data):
    session_id = data.get('session_id', None)
    df = get_data(session_id)

    if df is None:
        return {}

    comparisons = {category: {}}
    comparisons = create_comparisons(df, 'pay', 'gender', comparisons)
    fig = dumbbell_chart(comparisons, category, method, sort)

    return fig


dash.register_page(
    __name__,
    path=PATH,
    title=NAME,
    name=NAME,
    order=2,
    layout=layout,
    default=page_layout()
)
