
"""
File Name: page_2.py
Author: Connor McCaffrey
Date: 8/16/2024

Description:
    - This file contains the layout and callback functions used for the "Variable Selection" page.
"""

import dash
from dash import dcc, html, Input, Output, State, callback, ALL
import dash_bootstrap_components as dbc
from utils.cache import query_data, query_comparisons
from dash.exceptions import PreventUpdate

NAME = "Variable Selection"
PATH = "/variable-selection"

layout = html.Div()

response_selection = html.Div(className="pb-3", children=[
    dbc.Label("Response Variable", html_for="response-selection", className="input-group-label"),
    dcc.Dropdown(
        id="response-selection",
        multi=False,
        placeholder="Select continuous response variable",
        disabled=True,
        clearable=False,
        persistence=True,
        persistence_type="session",
    )
])


toggle_btn = dbc.Button("Select All", id="toggle-btn", className="my-3 py-2 mx-1 px-2", color="success", n_clicks=0)


quant_selection = html.Div(className="pb-3", children=[
    dbc.Label("Quantitative Predictor(s)", html_for="quant-selection", className="input-group-label"),
    dcc.Dropdown(
        id="quant-selection",
        multi=True,
        placeholder="Select quantitative predictor(s) of pay",
        disabled=True,
        persistence=True,
        value=[],
        persistence_type="session",
    )
])


cat_selection = html.Div(className="pb-3", children=[
    dbc.Label("Categorical Predictor(s)", html_for="cat-selection", className="input-group-label"),
    dcc.Dropdown(
        id="cat-selection",
        multi=True,
        placeholder="Select categorical predictor(s) of pay",
        disabled=True,
        persistence=True,
        persistence_type="session",
    )
])


base_selection = html.Div([
    dbc.Label("Base Levels", html_for="cat-selection", className="input-group-label"),
    html.Div(
        id="base-selections",
        style={
            'display': 'flex',
            'flexDirection': 'column',
            'gap': '10px',
            'max-height': '400px',
            "alignItems": "center",
            'overflowY': "auto",
            "border": "1px solid #ddd",
            "border-radius": "5px",
            "padding": "10px",
        },
    ),
])

selection_btn = html.Div(className="text-center", children=[
    dbc.Button("Submit Selection", id="selection-btn", className="my-3 py-2 mx-1 px-2", color="success", n_clicks=0)
])


def page_layout():
    layout = html.Div(id="layout", children=[
        dbc.Container([
            html.Div(className="plot-frame px-5 py-5", children=[
                response_selection,
                quant_selection,
                cat_selection,
                base_selection,
                selection_btn,
            ])
        ], className="my-4"),
    ])

    return layout


@callback(
    Output('response-selection', 'options'),
    Output("response-selection", "disabled"),
    Input('session', 'data'),
)
def update_response_options(data):
    session_id = data.get('session_id', None)

    df, _ = query_data(session_id)

    if df is not None:
        numeric = df.select_dtypes(include=["number"]).columns
        #opts = [{'label': c.capitalize(), 'value': c} for c in numeric]
        return numeric, False

    return [], True



@callback(
    Output("quant-selection", "options"),
    Output("quant-selection", "disabled"),
    Input("response-selection", "value"),
    State("response-selection", "options"),
    State("session", "data")

)
def update_quant_options(response, response_options, data):
    if response is None:
        return [], True

    opts = []

    if response_options is not None:
        opts = [c for c in response_options if c != response]
    else:
        session_id = data.get("session_id", None)
        df, _ = query_data(session_id)

        if df is not None:
            numeric = df.select_dtypes(include=["number"]).columns
            opts = [c for c in numeric if c != response]

    return opts, False


@callback(
    Output("cat-selection", "options"),
    Output("cat-selection", "disabled"),
    Input("response-selection", "value"),
    State("session", "data"),

)
def update_cat_options(response, data):
    if response is None:
        return [], True

    session_id = data.get('session_id', None)
    df, _ = query_data(session_id)

    if df is None:
        return [], True

    threshold = 20
    nonnumeric = df.select_dtypes(exclude=["number"]).columns
    nonnumeric = [c for c in nonnumeric if df[c].nunique() <= threshold]

    opts = [c for c in nonnumeric if c != response]

    return opts, False


@callback(
    Output("base-selections", "children"),
    Input("cat-selection", "value"),
    State("base-selections", "children"),
    State("session", "data"),
)
def update_base_levels(cats, base_selections, data):
    default = html.Div([html.P("No categorical variables selected")], className="pt-3")

    if not cats:
        return [default]

    session_id = data.get('session_id', None)
    df, _ = query_data(session_id)

    if df is None:
        return [default]

    result = []
    for c in cats:
        lvls = df[c].unique()
        lvl = df[c].value_counts().idxmax()


        dropdown = html.Div([
            dbc.Label(f"{c}", html_for=f"base-{c}", className="input-group-label px-2 mt-2 pt-1",  style={"height": "35px", 'width': '200px', "border": "1px solid #ddd", "border-radius": "5px"}),
            dcc.Dropdown(
                id={"type": "base-dropdown", "index": c},
                className="base-dropdown",
                placeholder=f"Select {c} base level",
                options=lvls,
                value=lvl,
                clearable=False,
                style={"width": "350px"},
                persistence=True,
                persistence_type="session",
            )
        ], style={'display': 'flex', 'alignItems': 'center', 'gap': '10px', "position": "relative"})

        result.append(dropdown)

    return result


@callback(
    Output("session", "data", allow_duplicate=True),
    Input("selection-btn", "n_clicks"),
    State("response-selection", "value"),
    State("quant-selection", "value"),
    State("cat-selection", "value"),
    State({"type": "base-dropdown", "index": ALL}, "value"),
    State("session", "data"),
    prevent_initial_call=True,
)
def submit_selection(n_clicks, response, quants, cats, base_selections, data):
    if n_clicks == 0:
        raise PreventUpdate

    if response is None:
        raise PreventUpdate

    if quants is None and cats is None:
        raise PreventUpdate

    session_id = data.get("session_id", None)
    valid_data = data.get("valid_data", False)

    if not valid_data:
        raise PreventUpdate

    categorical = {}
    if cats is not None:
        for i, c in enumerate(cats):
            try:
                base = base_selections[i]
                categorical[c] = {"base": base}
            except IndexError as err:
                print(f"Index Error while iterating through lists of potential base levels. Preventing submission. Message: {err}")
                raise PreventUpdate

    comps = {"response": response, "quantitative": quants, "categorical": categorical}
    comparisons = query_comparisons(session_id, comps) # timestamp

    if comparisons is None:
        data["valid_selection"] = False
    else:
        data["valid_selection"] = True

    return data


dash.register_page(
    __name__,
    path=PATH,
    title=NAME,
    name=NAME,
    order=2,
    layout=layout,
    default=page_layout(),
)
