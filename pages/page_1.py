
"""
File Name: page_1.py
Author: Connor McCaffrey
Date: 8/16/2024

Description:
    - This file contains the layout and callback functions used for the "Data Generation" page.
"""

import dash
import numpy as np
import pandas as pd
from dash import dcc, html, Input, Output, State, callback, dash_table
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
from utils.config import GRAPHCONFIG, BLUE, PINK, alpha, EMPTYFIG, AXISBLANK, MARGINBLANK
import plotly.graph_objs as go
import base64
import io
import datetime
from scipy.stats import norm
from utils.cache import query_data

NAME = "Data Generation"
PATH = "/data-generation"

# Gender Gap
gender_gap = html.Div([
    dbc.Label("Gender Gap", html_for="gender-gap", className="input-group-label"),
    html.Div([
        html.Div(html.Span("$", className="input-group-text"), className="input-group-prepend"),
        dbc.Input(id="gender-gap", type="number", placeholder=0, value=0, step=5000, persistence=True, className="form-control"),
        html.Div(html.Span(".00", className="input-group-text")),
    ], className="input-group")
], className="my-3")

# Gender Ratio
gender_ratio = html.Div([
    dbc.Label("Gender Ratio (%Male)", html_for="slider"),
    dcc.Slider(id="gender-ratio", persistence=True, min=0, max=1, value=0.5, marks={0: "0%", 0.25: "25%", 0.5: "50%", 0.75: "75%", 1: "100%"}),
], className="my-3")

sample_size = html.Div([
    dbc.Label("Sample Size", html_for="sample-size", className="input-group-label"),
    html.Div([
        html.Div(html.Span("N", className="input-group-text"), className="input-group-prepend"),
        dbc.Input(id="sample-size", type="number", placeholder=10000, value=10000, step=1000, persistence=True, className="form-control"),
    ], className="input-group")
], className="my-3")


# custom styles for input boxes
custom_styles = {
    'equal-height-columns': {
        'display': 'flex',
        'flexDirection': 'column',
        'justifyContent': 'stretch',
        'alignItems': 'stretch',
        'height': '100%'
    },
    'vertical-center': {
        'display': 'flex',
        'flexDirection': 'column',
        'justifyContent': 'center',
        'alignItems': 'center',
        'height': '100%'
    }
}

# File Upload
upload = html.Div([
    dcc.Upload(
        id="upload-data",
        children=html.Div(["Drag and Drop or Select Files"]),
        style={
            "width": "100%",
            "height": "100%",
            "lineHeight": "60px",
            "borderWidth": "1px",
            "borderStyle": "dashed",
            "borderRadius": "5px",
            "borderColor": "lightgray",
            "textAlign": "center",
        },
        multiple=False, # whether multiple files can be uploaded
        className="mx-30"
    ),
    html.Div(id="upload-display")
])

graph_preview =  html.Div([
    dcc.Graph(
        id="graph-preview",
        figure=EMPTYFIG,
        config=GRAPHCONFIG,
        style={'width': '530px', 'height': '300px'},
        className="m-30"
    )
], style={'display': 'flex', 'justify-content': 'center', 'align-items': 'center'})



# Form Creation
form = dbc.Form([
    dbc.Container([
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.H4("Specify Parameters", style={'textAlign': 'center'}),
                    gender_gap,
                    sample_size,
                    gender_ratio,
                    graph_preview,
                ], id="input1", className="p-3 border rounded", style=custom_styles['equal-height-columns']),
            ], width=5),
            dbc.Col([
                html.Div([
                    html.H4(" or ", className="text-center my-4", style=custom_styles['vertical-center']),
                ]),
            ], width=2, className="d-flex align-items-center justify-content-center"),
            dbc.Col([
                html.Div([
                    html.H4("Upload Data File", style={'textAlign': 'center'}),
                    upload,
                ], id="input2", className="p-3 border rounded", style=custom_styles['equal-height-columns']),
            ], width=5),
        ], className="mt-4"),
    ], className="my-4"),
    html.Div([
       dbc.Stack([
            html.Div(children=[
                dbc.Button("Advanced", id="advanced-button", className="my-3 py-2 mx-1 px-2", color="primary"),
                dbc.Button("Generate Data", id="submit-btn", className="my-3 py-2 mx-1 px-2", color="success", n_clicks=0),
            ], className="mx-auto"),
        ], direction="horizontal", className="g-0"),
    ]),
])

alert = dbc.Container([
    dbc.Alert(
        "New Data Generated Successfully!",
        id="alert",
        dismissable=False,
        is_open=False,
        duration=3000,
        fade=True,
        className="mx-auto",
    ),
])

layout = html.Div()

def page_layout():
    layout = html.Div(id="layout", children=[
        form,
        alert,
    ])
    return layout

@callback(
    Output('session', 'data', allow_duplicate=True),
    Input("submit-btn", "n_clicks"),
    State("gender-gap", "value"),
    State("gender-ratio", "value"),
    State("sample-size", "value"),
    State("upload-data", "contents"),
    State("upload-data", "filename"),
    State("session", "data"),
    prevent_initial_call=True,
)
def save_form(n_clicks, gender_gap, gender_ratio, sample_size, upload, filename, data):
    if n_clicks == 0:
        raise PreventUpdate

    session_id = data.get('session_id', None)
    params = {"gender_gap": gender_gap, "gender_ratio": gender_ratio, "sample_size": sample_size}
    df, timestamp = query_data(session_id, params=params, upload=upload, filename=filename)

    if df is None:
        data["valid_data"] = False
    else:
        data["valid_data"] = True
        data['timestamp'] = timestamp

    return data




@callback(
    Output("alert", "is_open"),
    Input("submit-btn", "n_clicks"),
    State("alert", "is_open"),
)
def toggle_alert(n, is_open):
    if n:
        return not is_open
    return is_open


@callback(
    Output('graph-preview', 'figure'),
    Input('gender-gap', 'value'),
    Input('gender-ratio', 'value'),
    State("graph-preview", "figure")
)
def update_shape(gap, ratio, fig):
    if gap is None or ratio is None:
        return fig

    xmin = 0
    xmax = 200000
    xmean = (xmin + xmax) / 2
    sd = (xmax - xmin) / 8
    x = np.linspace(xmin, xmax, 500)
    y_m = norm.pdf(x, xmean + gap/2, sd) * ratio
    y_f = norm.pdf(x, xmean - gap/2, sd) * (1-ratio)

    # Create plotly figure object
    fig = go.Figure()

    # Add PDF curve trace
    fig.add_trace(go.Scatter(
        x=x,
        y=y_m,
        mode='lines',
        line=dict(width=3, color=BLUE),
        fill='tozeroy',
        fillcolor=alpha(BLUE, 0.2),
        showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=x,
        y=y_f,
        mode='lines',
        line=dict(width=3, color=PINK),
        fill='tozeroy',
        fillcolor=alpha(PINK, 0.2),
        showlegend=False,
    ))

    fig.layout = go.Layout(xaxis=AXISBLANK, yaxis=AXISBLANK, margin=MARGINBLANK)
    fig.update_layout(transition_duration=500,
                      template="simple_white",
                      xaxis_title="Simulated Pay Gap",
                      xaxis_range=[xmin, xmax],
                      width=520,
                      height=300,
                      margin=dict(l=30, r=30, t=30, b=30))
    return fig



def parse_contents(contents, filename, date):
    content_type, content_string = contents.split(",")
    decoded = base64.b64decode(content_string)
    try:
        if "csv" in filename:
            df = pd.read_csv(io.StringIO(decoded.decode("utf-8")))
        elif "xls" in filename:
            df = pd.read_excel(io.BytesIO(decoded))
    except Exception as e:
        return html.Div([f"There was an error processing the file: {e}"])
    else:
        return html.Div([
            html.Hr(),
            html.H5(filename),
            html.H6("Last modified " + str(datetime.datetime.fromtimestamp(date))),
            dash_table.DataTable(
                data=df.to_dict("records"),
                columns=[{"name": i, "id": i} for i in df.columns],
                #page_size=10,
                fixed_rows={'headers': True},
                fixed_columns={'headers': True, 'data':1},
                style_table={'height': "450px"},
                style_cell={"font-family": "Gotham Thin"},
            ),
            html.Hr(),
        ], className="p-30")


@callback(
    Output("upload-display", "children"),
    Input("upload-data", "contents"),
    [State("upload-data", "filename"),
     State("upload-data", "last_modified")],
)
def update_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        children = parse_contents(list_of_contents, list_of_names, list_of_dates)
        return children



dash.register_page(
    __name__,
    path=PATH,
    title=NAME,
    name=NAME,
    order=1,
    layout=layout,
    default=page_layout(),
)
