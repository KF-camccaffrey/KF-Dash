
import dash
import numpy as np

dash.register_page(__name__)

from dash import Dash, dcc, html, Input, Output, State, callback, callback_context
import plotly.express as px
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
from dash.dependencies import MATCH, ALL

# Gender Gap
gender_gap = dbc.FormFloating([
    dbc.Input(id="gender-gap", type="number", placeholder="10000"),
    dbc.Label("Gender Gap ($)"),
])

# Male Pay Range
male_pay_range = html.Div([
    dbc.Label("Male Pay Range", html_for="range-slider"),
    dcc.RangeSlider(id="male-pay-range", persistence=True, min=0, max=100000, value=[60000,80000]),
], className="my-3")

# Female Pay Range
female_pay_range = html.Div([
    dbc.Label("Female Pay Range", html_for="range-slider"),
    dcc.RangeSlider(id="female-pay-range", persistence=True, min=0, max=100000, value=[60000,80000]),
], className="my-3")

# Gender Ratio
gender_ratio = html.Div([
    dbc.Label("Gender Ratio", html_for="slider"),
    dcc.Slider(id="gender-ratio", persistence=True, min=0, max=100, value=50),
], className="my-3")

# Male Advanced
male_advanced = dbc.Row([
    dbc.Label("Mean", width="auto"),
    dbc.Col(
        dbc.Input(id="male-mean", type="number", placeholder="0"),
        className="me-3"
    ),

    dbc.Label("StD", width="auto"),
    dbc.Col(
        dbc.Input(id="male-std", type="number", placeholder="1"),
        className="me-3"
    ),

    dbc.Label("Sample", width="auto"),
    dbc.Col(
        dbc.Input(id="male-n", type="number", placeholder="1000"),
        className="me-3"
    ),
])

# Female Advanced
female_advanced = dbc.Row([
    dbc.Label("Mean", width="auto"),
    dbc.Col(
        dbc.Input(id="female-mean", type="number", placeholder="0"),
        className="me-3"
    ),

    dbc.Label("StD", width="auto"),
    dbc.Col(
        dbc.Input(id="female-std", type="number", placeholder="1"),
        className="me-3"
    ),

    dbc.Label("Sample", width="auto"),
    dbc.Col(
        dbc.Input(id="female-n", type="number", placeholder="1000"),
        className="me-3"
    ),
])

# Random Seed
random_seed =  dbc.FormFloating([
    dbc.Input(id="random-seed", type="number", placeholder="42"),
    dbc.Label("Random Seed"),
])

# Form Creation
form = dbc.Form([
    dbc.Container([
        male_pay_range,
        female_pay_range,
        gender_ratio,
    ]),

    html.Div([
        dbc.Collapse(
            html.Div([
                male_advanced,
                female_advanced,
                random_seed,
                gender_gap,
            ]),
            id="advanced",
            is_open=False
        ),
        dbc.Stack([
            html.Div(className="mx-auto", children=[
                dbc.Button("Advanced", id="advanced-button", className="my-3 py-2 mx-1 px-2", color="primary"),
                dbc.Button("Generate Data", id="submit-btn", className="my-3 py-2 mx-1 px-2", color="success"),
            ])
        ], direction="horizontal", className="g-0")
    ]),
])


layout = html.Div([
    dcc.Store(id='session', storage_type='session'),
    form,
])


@callback(
    Output("advanced", "is_open"),
    [Input("advanced-button", "n_clicks")],
    [State("advanced", "is_open")],
)
def toggle_collapse(n, is_open):
    if n:
        return not is_open
    return is_open


@callback(
    Output('session', 'data', allow_duplicate=True),
    Input("submit-btn", "n_clicks"),
    [
        State("gender-gap", "value"),
        State("male-pay-range", "value"),
        State("female-pay-range", "value"),
        State("gender-ratio", "value"),
        State("male-mean", "value"),
        State("male-std", "value"),
        State("male-n", "value"),
        State("female-mean", "value"),
        State("female-std", "value"),
        State("female-n", "value"),
        State("random-seed", "value"),
        State("session", "data")
    ],
    prevent_initial_call=True
)
def save_form(n_clicks, gender_gap, male_pay_range, female_pay_range,
              gender_ratio, male_mean, male_std, male_n,
              female_mean, female_std, female_n, random_seed, data):
    data = data or {
        #'gender_gap': 10000,
        'male_pay_range': [60000, 80000],
        'female_pay_range': [60000, 80000],
        "gender_ratio": 50,
    }

    if n_clicks is None:
        return data

    #data["gender_gap"] = gender_gap
    data["male_pay_range"] = male_pay_range
    data["female_pay_range"] = female_pay_range
    data["gender_ratio"] = gender_ratio
    return data
