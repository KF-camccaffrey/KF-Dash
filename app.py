
"""
File Name: app.py
Author: Connor McCaffrey
Date: 8/16/2024

Description:
    - This file is the main dash app entrypoint.
    - It defines the overall layout for the app (header, tabs, footer, etc.)
    - It contains callback functions for managing the URL and session data
"""

import dash
from dash import html, dcc, Output, Input, State, callback
import datetime
import dash_bootstrap_components as dbc
import uuid
from utils.config import styled

# initialize dash app
external_stylesheets = [
    dbc.themes.BOOTSTRAP,
    "/assets/styles.css"
]

app = dash.Dash(__name__, use_pages=True, external_stylesheets=external_stylesheets)
app.config.suppress_callback_exceptions = True

# this import is defined below "app" to avoid circular imports
from utils.cache import query_data, query_comparisons, query_model, clear_cache

PAGES = [page for page in dash.page_registry.values()]
PATHS, ORDERS, NAMES = [list(page[key] for page in PAGES) for key in ["path", "order", "name"]]
MAXPAGE = ORDERS[-1]

tabs = html.Div(dbc.Nav([
    dbc.NavItem(dbc.NavLink(page["name"], href=page["relative_path"], active="exact"), className="li", id=f"tab-{page['order']}")
    for page in dash.page_registry.values()
    if page["module"] not in ["pages.home", "pages.not_found_404"]
], id="tabs", justified=True, className="ol"), style={"margin": "0 auto", "width": "65%"})

header = dbc.Navbar(id="navbar", fixed="top", color="white", className="border-bottom", children=[
    dbc.Container(fluid=True, children=[
        dbc.NavLink(html.Img(src="/assets/images/logo.png", height="30px"), href="https://www.kornferry.com/"),
        dbc.NavbarBrand("Pay Equity Dashboard Demo", id="nav-title", className="mx-5", href="/"),
        tabs,
        dbc.Button("♻", id="recycle-btn", size="sm", color="light", style={"border-radius": 20}, n_clicks=0),
    ])
])


footer = html.Footer(id="footer", className="d-flex flex-wrap justify-content-between align-items-center px-4 border-top", children=[
    html.P("© 2024 Korn Ferry", className="col-md-4 mb-0 text-light"),
    dbc.NavLink(html.Img(src="/assets/images/github_dark.png", height="30px"),
                href="https://github.com/KF-camccaffrey/KF-Dash",
                className="col-md-4 d-flex align-items-center justify-content-center mb-3 pt-3 link-dark text-decoration-none"),
    dbc.ListGroup(className="nav col-md-4 mb-0 justify-content-end", horizontal=True, children=[
        dbc.NavLink("Home", href="", className="nav-link px-2 text-light"),
        dbc.NavLink("Features", href="", className="nav-link px-2 text-light"),
        dbc.NavLink("Pricing", href="", className="nav-link px-2 text-light"),
        dbc.NavLink("About", href="", className="nav-link px-2 text-light")
    ])
])


app.layout = html.Div([
    dcc.Location(id="url", refresh=False),
    dcc.Store(id='session', storage_type='session'),
    dbc.Container(id="app-container", fluid=True, children=[
        dbc.Row(header, id="app-row1"),
        dbc.Row(id="app-row2", children=[
            html.Div(dash.page_container, id="page-content")
        ]),
        dbc.Row(footer, id="app-row3"),
    ])
])


@callback(
    Output("session", "data"),
    Output("page-content", "children"),
    [Output(f'tab-{i}', 'className') for i in range(1, MAXPAGE+1)],
    Input("url", "pathname"),
    State("session", "data"),
)
def handle_url(path, data):
    data = update_session(data)
    try:
        i = PATHS.index(path)
    except ValueError:
        i = 0
    finally:
        output_session(path, **data)
        content = validate_url(i, data["valid_data"], data["valid_selection"])
        names = update_tabs(i)
    return data, content, *names


def update_session(data):
    data = data if data else {}

    session_id = data.get("session_id", str(uuid.uuid4()))
    timestamp = data.get("timestamp", datetime.datetime.now())
    valid_data = data.get("valid_data", False)
    valid_selection = data.get("valid_selection", False)
    valid_model = data.get("valid_model", False)

    valid_data = query_data(session_id, check=True) if valid_data else False
    valid_selection = query_comparisons(session_id, check=True) if valid_selection else False # timestamp
    valid_selection = valid_selection if valid_data else False
    valid_model = query_model(session_id, check=True) if valid_model else False # timestamp
    valid_model = valid_model if valid_selection else False


    data = {"session_id": session_id, "timestamp": timestamp, "valid_data": valid_data, "valid_selection": valid_selection, "valid_model": valid_model}
    return data

def output_session(path, session_id, timestamp, valid_data, valid_selection, valid_model):
    message = f"""{styled("SESSION:", "black", True)}
    {styled("PATH:", "red",     True)}\t{styled(path,            "red")}
    {styled("UUID:", "magenta", True)}\t{styled(session_id,      "magenta")}
    {styled("TIME:", "blue",    True)}\t{styled(timestamp,       "blue")}
    {styled("DATA?", "cyan",    True)}\t{styled(valid_data,      "cyan")}
    {styled("VARS?", "green",   True)}\t{styled(valid_selection, "green")}
    {styled("MODL?", "yellow",  True)}\t{styled(valid_model,     "yellow")}\n"""
    print(message)
    return


def validate_url(i, valid_data, valid_selection):
    o = i - 1
    if o > 1 and not valid_data:
        return no_data_layout()
    elif o > 2 and not valid_selection:
        return no_selection_layout()
    else:
        return PAGES[i]["default"]

def no_data_layout():
    layout = dbc.Container(
        dbc.Row(
            dbc.Col(
                html.Div([
                    html.H2("No Data Available"),
                    html.P("Sorry, a dataset must be loaded before viewing this page."),
                    dbc.Button("Go to Data Generation", color="primary", href="/data-generation"),
                ], style={"text-align": "center", "margin-top": "20%"}),

                width={"size": 6, "offset": 3},  # Center the column
            ),
        ),
        className="h-100",  # Make sure the container takes full height
    )
    return layout

def no_selection_layout():
    layout = dbc.Container(
        dbc.Row(
            dbc.Col(
                html.Div([
                    html.H2("No Variables Selected"),
                    html.P("Sorry, a response variable must be selected before viewing this page."),
                    dbc.Button("Go to Variable Selection", color="primary", href="/variable-selection"),
                ], style={"text-align": "center", "margin-top": "20%"}),

                width={"size": 6, "offset": 3},  # Center the column
            ),
        ),
        className="h-100",  # Make sure the container takes full height
    )
    return layout


def update_tabs(i):
    o = i - 1
    nA = 1 if o > 0 else 0
    nL = o - 1
    nR = MAXPAGE - o if o >= 0 else MAXPAGE

    names = ["li left-of-active"] * nL + ["li"] * nA + ["li right-of-active"] * nR
    return names


@callback(
    Input("recycle-btn", "n_clicks"),
)
def empty_cache(n_clicks):
    if n_clicks > 0:
        clear_cache()
    return



if __name__ == "__main__":
    app.run_server(debug=True)
