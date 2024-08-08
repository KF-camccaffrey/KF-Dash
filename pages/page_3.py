

import dash
from dash import Dash, dcc, html, Input, Output, State, callback, dash_table, ctx
from utils import generator
from utils.config import BLUE, PINK, FORESTGREEN, GRAPHCONFIG, HOVERLABEL, BASICCOMPS, EMPTYFIG
import os
from dash.exceptions import PreventUpdate

import plotly.express as px
import plotly.figure_factory as ff
import numpy as np
import pandas as pd
import dash_bootstrap_components as dbc
from utils.cache import query_data, query_comparisons

from utils.comparisons import create_comparisons, dumbbell_chart, pie_chart, metric_bars

NAME = "Visualizations"
PATH = "/visualizations"

data_graph = dcc.Graph(figure=EMPTYFIG, id='generate-chart', config=GRAPHCONFIG)
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
        figure=EMPTYFIG,
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
    figure=EMPTYFIG,
    config=GRAPHCONFIG,
    className="px-3 py-3"
)

boxplots = dcc.Graph(
    figure=EMPTYFIG,
    id='boxplot',
    config=GRAPHCONFIG,
    className="px-3 py-3"
)

piechart = dcc.Graph(
    figure=EMPTYFIG,
    id='piechart',
    config=GRAPHCONFIG,
    className="px-3 py-3"
)

caption = html.Div(id="caption", className="px-3 py-3")
level = html.Div(id="level", hidden=True)
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
                    level,
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

    #print("query_data in update_chart()")
    df, _ = query_data(session_id)

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

    #print("query_data in update_table()")
    df, _ = query_data(session_id)
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
    #print("query_data in update_boxplot()")
    session_id = data.get('session_id', None)
    df = query_data(session_id)

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
    Input('level', 'children'),
    State('session', 'data'),
)
def update_boxplot(category, lvl, data):
    session_id = data.get('session_id', None)
    df, _ = query_data(session_id)

    if df is None:
        return EMPTYFIG

    if category is None:
        return EMPTYFIG

    if category == "none":
        return EMPTYFIG

    if lvl != "Overall":
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
    Output("level", "children"),
    Input('category-input', 'value'),
    Input('dumbbell', 'hoverData'),
    Input('dumbbell', 'clickData'),
)
def update_level(category, hoverData, clickData):
    if clickData is None and hoverData is None:
        lvl = "Overall"
    else:
        if clickData is None:
            clickData = hoverData

        lvl = clickData["points"][0]['y']
        if ctx.triggered_id == "category-input":
            lvl = "Overall"
    return lvl


@callback(
    Output('dumbbell', 'clickData'),
    Input('category-input', 'value')
)
def reset_dumbbell_data(category):
    return None



@callback(
    Output('metrics', 'children'),
    Input('category-input', 'value'),
    Input('level', 'children'),
    Input('method-input', 'value'),
    State('session', 'data'),
)
def update_metrics(category, lvl, method, data):
    session_id = data.get('session_id', None)
    timestamp = data.get('timestamp', None)
    comparisons = query_comparisons(session_id, timestamp)

    if comparisons is None:
        return [html.H5("Level Summary"), html.H4("No Data Available")]

    if category is None:
        return [html.H5("Level Summary"), html.H4("No Data Available")]

    categorical = comparisons["categorical"]

    metrics = categorical.get(category, None)

    if metrics is None:
        return [html.H5("Level Summary"), html.H4("No Data Available")]

    lvls = metrics["levels"]
    i = np.where(lvls == lvl)[0][0]
    keys = [f"{method}_male", f"{method}_female", f"{method}_other", f"{method}_gap", "n_male", "n_female", f"{method}_gap_perc"]
    metric_male, metric_female, metric_other, metric_gap, n_male, n_female, ratio = (metrics[key][i] for key in keys)
    metric_max = max([metric_male, metric_female, metric_other])

    fig_pie = pie_chart(comparisons, category, lvl)
    fig_bar = metric_bars(comparisons, category, lvl, method)

    if method == "mean":
        top = "48%"
    else:
        top = "54%"

    layout = [
        html.H5("Level Summary"),
        html.H4(lvl),
        html.Div(id="metric-display", className="mx-2", children=[
            dbc.Row([
            #dbc.Col(width=8, children=[
            #html.H5(className="pt-4", children=[
            #    html.Div("Avg. Men's Pay:", className="metric-label blue"),
            #    html.Div('${:20,.2f}'.format(metric_male), className="metric-value tabular blue"),
            #    #html.Div("N:", className="n-label blue"),
            #    #html.Div('{:20}'.format(n_male),  className="n-value tabular blue")
            #]),
            #html.H5(children=[
            #    html.Div("Avg. Women's Pay:", className="metric-label pink"),
            #    html.Div('${:20,.2f}'.format(metric_female), className="metric-value tabular pink"),
            #    #html.Div("N:", className="n-label pink"),
            #    #html.Div('{:20}'.format(n_female),  className="n-value tabular pink")
            #]),
            #html.Hr(style={'margin': '10px 0', "width": "330px", "border-top": "2px solid gray"}),
            #html.H5(className="pb-4", children=[
            #    html.Div("Wage Gap:", className="metric-label"),
            #    html.Div('${:20,.2f}'.format(abs(metric_gap)), className="metric-value tabular")
            #]),
            #]),

            dbc.Col(width=1, children=[
                html.Div(
                    method.capitalize(),
                    style={
                        'transform': 'rotate(-90deg)',
                        'transform-origin': 'center center',
                        "top": top,
                        "left": "50%",
                        "z-index": 2,
                        "position": "relative",
                        "font-size": "20px",
                    }
                ),
            ]),
            dbc.Col(width=7, children=[
                html.Div(className="mt-2", children=[
                    create_bar(metric_male, metric_max, BLUE),
                    create_bar(metric_female, metric_max, PINK),
                    create_bar(metric_other, metric_max, FORESTGREEN),
                ])
            ]),



            dbc.Col(width=4, children=[
                dcc.Graph(
                    figure=fig_pie,
                    config=GRAPHCONFIG,
                    className="px-3"
                ),
            ])
            ]),
        ]),

        #dcc.Graph(
        #            figure=fig_bar,
        #            config=GRAPHCONFIG,
        #            className="px-3"
        #),

        html.Div(className="mt-2 py-2", style={"textAlign": "center", "border": "2px solid #DEE2E6", "border-radius": "15px"}, children=[
            "A woman makes ",
            html.Span("${:.2f}".format(ratio), className="tabular pink"),
            " for every ",
            html.Span("$1.00", className="tabular blue"),
            " a man makes."
        ])
    ]

    return layout


# Define a function to create a bar with a width proportional to the value
def create_bar(value, max_value, color, text_padding_left=20):
    height = "39px"
    width = "300px"

    perc = 100 *  value / max_value
    if perc < 25:
        textColor = "gray"
    else:
        textColor = "white"

    layout = html.Div(children=[
        # Bar Container
        html.Div(style={"position": "relative", "height": height, "width": width}, children=[
            html.Div(
                className="metric-label mt-1",
                children='${:20,.2f}'.format(value),
                style={
                    'position': 'absolute',
                    'left': f'{text_padding_left}px',  # Distance from the left end of the bar
                    #'right': '0',  # Align text to the right within the padding
                    'color': textColor,
                    "width": "150px",
                    "font-size": 20,
                    'text-align': 'right',
                    #'padding': '5px',
                    'height':  height,
                    "z-index": 1,
                }
            ),
            html.Div(className="mt-1", style={
                'backgroundColor': color,
                'width': f'{perc}%',  # Percentage width
                'height': height,
                #'marginBottom': '10px',
                #'position': 'relative',  # Positioning context for the text
                #'color': 'white',
                #'overflow': 'hidden',  # Ensure no overflow is visible
                "border": "2px solid #DEE2E6",
                "border-radius": "3px",
                "opacity": 1.0,
                "z-index": 2,
            }),
        ])
    ])

    return layout







@callback(
    Output('piechart', 'figure'),
    Input('category-input', 'value'),
    Input('level', 'children'),
    State('session', 'data'),
)

def update_piechart(category, lvl, data):
    session_id = data.get('session_id', None)
    timestamp = data.get('timestamp', None)
    comparisons = query_comparisons(session_id, timestamp)

    if comparisons is None:
        return {}

    fig = pie_chart(comparisons, category, lvl)
    return fig


@callback(
    Output('caption', 'children'),
    Input('category-input', 'value'),
    Input('method-input', 'value'),
)
def update_caption(category, method):
    if category is None:
        label = f"No Data Available"
    elif category == "none":
        label = f"No Data Available"
    else:
        label = f"By {category.capitalize()}"

    layout = [
        html.H5(f"Difference in {method.capitalize()} Pay Between Men & Women"),
        html.H4(label)
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
    timestamp = data.get('timestamp', None)
    comparisons = query_comparisons(session_id, timestamp)

    if comparisons is None:
        return EMPTYFIG

    categorical = comparisons["categorical"]
    metrics = categorical.get(category, None)

    if metrics is None:
        return EMPTYFIG

    fig = dumbbell_chart(comparisons, category, method, sort)
    return fig


@callback(
    Output("category-input", "value"),
    Output("category-input", "options"),
    Input("session", "data"),
    State("category-input", "value"),
)

def set_details_by(data, category):
    session_id = data.get("session_id", None)
    timestamp = data.get("timestamp", None)

    comparisons = query_comparisons(session_id, timestamp)

    if comparisons is None:
        return "none", [{"label": " No Data Available", "value": "none"}]

    categorical = comparisons["categorical"]
    keys = list(categorical.keys())
    options = [{"label": f" {k.capitalize()}", "value": k} for k in keys if not all(np.isnan(categorical[k]["mean_gap"][1:]))]

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
    order=3,
    layout=layout,
    default=page_layout()
)
