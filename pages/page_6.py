
"""
File Name: page_6.py
Author: Connor McCaffrey
Date: 8/16/2024

Description:
    - This file contains the layout and callback functions used for the "Multivariate Regression" page.
"""

import dash
from dash import dcc, html, Input, Output, State, callback, ALL
import dash_ag_grid as dag
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
from utils.cache import query_data, query_comparisons, query_model
from utils.config import FORESTGREEN, GRAPHCONFIG, EMPTYFIG, alpha, RED, GRAY
import statsmodels.api as sm
import scipy.stats as stats
import plotly.graph_objects as go
from dash.exceptions import PreventUpdate

NAME = "Multivariate Regression"
PATH = "/multivariate-regression"

layout = html.Div()

trans_selection = dcc.RadioItems(
    id="trans-selection",
    options=[
        {'label': ' None', 'value': 'none'},
        {'label': ' Log', 'value': 'log'},
    ],
    value='none',
    labelStyle={"display": "inline-block", "padding": "5px"}
)

quant_pred = dcc.Dropdown(
    id="quant-pred",
    multi=True,
    placeholder="Select quantitative predictor(s) of pay",
    persistence=True,
    persistence_type="session",
)

cat_pred = dcc.Dropdown(
    id="cat-pred",
    multi=True,
    placeholder="Select categorical predictor(s) of pay",
    persistence=True,
    persistence_type="session",
)

dummy_selection = dcc.Dropdown(
    id="dummy-selection",
    multi=True,
    placeholder="Select dummy variables",
)

dummy_pred = html.Div(
    id="dummy-pred",
    style={
        'display': 'flex',
        'flexDirection': 'column',
        'gap': '10px',
        'height': '300px',
        "alignItems": "center",
        'overflowY': "auto",
        "border": "1px solid #ddd",
        "background": "white",
        "border-radius": "5px",
        "padding": "10px",
    },
)

alpha_slider = html.Div(className="pt-3", children=[
    dbc.Label(f"Significance Level (α):", className="input-group-label"),
    dcc.Slider(id="alpha-slider", persistence=True, min=1, max=6, value=4, step=1, marks={1: "0.1%", 2: "1%", 3: "2.5%", 4: "5%", 5: "10%", 6: "20%"})
])

res_type = html.Div(className="pt-3 px-3", children=[
    dbc.Label(f"Plot Type:", className="input-group-label"),
    dcc.Dropdown(
        id="res-type",
        persistence=True,
        persistence_type="session",
        options=[
            {"label": "Actual vs. Expected Pay", "value": 1},
            {"label": "Residuals", "value": 2},
            {"label": "Standardized Residuals", "value": 3},
            {"label": "Studentized Residuals", "value": 4},
            {"label": "Normal Q-Q of Residuals", "value": 5},
            {"label": "Cook's Distance", "value": 6},
        ],
        value=1
    )
])

extra_options = dcc.Checklist(
    id="extra-options",
    options=[
       {'label': ' Hide Intercept', 'value': 0},
    ],
    inline=True,
    value=[0],
    persistence=True,
    persistence_type="session",
)


model_btn = dbc.Button("Build Model", id="model-btn", className="my-3 py-2 mx-1 px-2", color="success", n_clicks=0)

coeff_plot = dcc.Graph(
    id="coeff-plot",
    config=GRAPHCONFIG,
    figure=EMPTYFIG,
    className="px-3 py-3",
)

coeff_caption = html.P("No Data Available", id="coeff-caption", className="mx-4", style={"text-align": "center", "border": "2px solid #DEE2E6", "border-radius": "15px", "width": "700px"})

res_plot = dcc.Graph(
    id="res-plot",
    config=GRAPHCONFIG,
    figure=EMPTYFIG,
    className="px-3 py-3",
)

info_box = html.Div([
    html.Div(id="info-body", className="px-3 py-3"),
])

sideinputs = html.Div(className="pl-5", children=[
    html.Div(className="pb-2", children=[
        dbc.Label(html.H6("Response Transformation"), width="auto"),
        trans_selection,
    ]),
    html.Div(className="pb-2", children=[
        dbc.Label(html.H6("Quantitative Predictors"), width="auto"),
        quant_pred,
    ]),
    html.Div(className="pb-2", children=[
        dbc.Label(html.H6("Categorical Predictors"), width="auto"),
        cat_pred,
    ]),
    html.Div(className="pb-2", children=[
        dbc.Label(html.H6("Dummy Selection"), width="auto"),
        dummy_pred,
    ]),
    html.Div(className="pb-2", children=[
        dbc.Label(html.H6("More Options"), width="auto"),
        extra_options,
    ]),
])

def page_layout():
    layout = dbc.Container(id="layout", fluid=True, style={"height": "100%", "padding": "0px"}, children=[
        dbc.Row(style={"height": "100%"}, children=[
            dbc.Col(id="sidebar", width=3, children=[
                sideinputs,
            ]),
            dbc.Col(width=9, style={"height": "100%"}, children=[
                dbc.Row(style={"height": "65%"}, children=[
                    dbc.Col(id="plot-main", width=6, children=[
                        html.Div(className="plot-frame", children=[
                            dbc.Row([
                                dbc.Col(width=7, children=[
                                    html.H5("OLS Beta Estimates", className="px-3 pt-3"),
                                    html.H4(className="px-3", children=[
                                        html.Span("95%", style={"color": FORESTGREEN}, className="tabular", id="confidence"),
                                        html.Span(" Confidence Intervals"),
                                    ]),
                                ]),
                                dbc.Col(width=5, children=[
                                    alpha_slider,
                                ]),
                            ]),
                            dbc.Row(coeff_plot),
                            dbc.Row(coeff_caption)
                        ])
                    ]),
                    dbc.Col(id="plot-side", width=6, children=[
                        html.Div(className="plot-frame", children=[
                            dbc.Row([
                                dbc.Col(width=7, children=[
                                    html.H5("Model Diagnostics", className="px-3 pt-3"),
                                    html.H4("Residual Plot", id="res-plot-title", className="px-3"),
                                ]),
                                dbc.Col(width=5, children=[
                                    res_type,
                                ])
                            ]),
                            dbc.Row(res_plot),
                        ])
                    ]),
                ]),
                dbc.Row(style={"height": "35%"}, children=[
                    dbc.Col(id="plot-bottom", width=12, children=[
                        html.Div(className="plot-frame", children=[
                            info_box,
                        ])
                    ])
                ]),
            ]),
        ])
    ])

    return layout


@callback(
    Output('quant-pred', 'options'),
    Output('cat-pred', 'options'),
    Input('session', 'data'),
)
def update_options(data):
    session_id = data.get('session_id', None)
    comparisons = query_comparisons(session_id)

    if comparisons is None:
        return []

    quantitative = comparisons.get("quantitative", [])
    categorical = comparisons.get("categorical", {})
    return quantitative, list(categorical)


@callback(
    Output("dummy-pred", "children"),
    Input("cat-pred", "value"),
    State("session", "data"),
)
def update_dummies(predictors, data):
    default = html.Div([html.P("No categorical variables selected")], className="pt-3")

    if not predictors:
        return [default]

    session_id = data.get('session_id', None)
    comparisons = query_comparisons(session_id)

    if comparisons is None:
        return []

    categorical = comparisons.get("categorical", None)

    if categorical is None:
        return []

    result = []
    for c in predictors:
        metrics = categorical[c]
        lvls = metrics["levels"]
        dummy_idx = metrics["dummies"]
        dummies = lvls[dummy_idx]

        dropdown = html.Div([
            dbc.Label(f"{c}", className="input-group-label px-4 mt-2 pt-1",  style={"height": "35px", "width": "150px", "border": "1px solid #ddd", "border-radius": "5px"}),
            dcc.Dropdown(
                id={"type": "dummy-dropdown", "index": c},
                className="base-dropdown",
                placeholder=f"Select {c} dummy variables",
                multi=True,
                options=list(dummies),
                value=list(dummies),
                clearable=True,
                style={"width": "250px"},
                persistence=True,
                persistence_type="session",
            )
        ], style={'display': 'flex', 'alignItems': 'center', 'gap': '10px', "position": "relative"})

        result.append(dropdown)

    return result


@callback(
    Output("session", "data", allow_duplicate=True),
    Input("quant-pred", "value"),
    Input("cat-pred", "value"),
    Input({"type": "dummy-dropdown", "index": ALL}, "value"),
    Input("trans-selection", "value"),
    State("session", "data"),
    prevent_initial_call=True,
)
def save_model(quant_preds, cat_preds, dummy_lists, trans, data):
    session_id = data.get('session_id', None)
    df, _ = query_data(session_id)

    if df is None:
        raise PreventUpdate

    comparisons = query_comparisons(session_id) # timestamp

    if comparisons is None:
        raise PreventUpdate

    response = comparisons["response"]

    cat_dict = {}
    if cat_preds is not None:
        for i, c in enumerate(cat_preds):
            try:
                dummies = dummy_lists[i]
                if dummies:
                    cat_dict[c] = dummies
            except IndexError as err:
                print(f"Index Error while iterating through lists of dummy variables. Returning empty plots. Message: {err}")
                raise PreventUpdate

    if not quant_preds and not cat_dict:
        raise PreventUpdate

    y, X = preprocess(df, quant_preds, cat_dict, response, trans)
    model = query_model(session_id, y, X)

    if model is None:
        data["valid_model"] = False
    else:
        data["valid_model"] = True

    return data


def preprocess(df, quant_preds, cat_dict, response, trans):
    X = df[quant_preds]
    for cat in cat_dict:
        df_dummies = pd.get_dummies(df[cat], prefix=cat).astype(int)
        dummies = cat_dict[cat]
        df_dummies = df_dummies[[f"{cat}_{d}" for d in dummies]]
        X = pd.concat([X, df_dummies], axis=1)

    X = X.dropna()
    X = sm.add_constant(X)

    y = df[response][X.index]
    if trans == "log":
        c = 0.001
        y = np.where(y > 0, np.log(y), c)

    return y, X


@callback(
    Output("coeff-plot", "figure"),
    Input("extra-options", "value"),
    Input("alpha-slider", "value"),
    Input("session", "data"),
)
def plot_coeff(extra, alpha, data):
    session_id = data.get("session_id", None)
    model = query_model(session_id)

    if model is None:
        return EMPTYFIG

    n = model["n"]
    p = model["p"]
    pred = model["pred"]
    beta = model["beta"]
    bse = model["bse"]

    # remove intercept
    if 0 in extra:
        pred = pred[1:]
        beta = beta[1:]
        bse = bse[1:]
    else:
        pred[0] = "Intercept"

    match alpha:
        case 1:
            alpha = 0.001
        case 2:
            alpha = 0.01
        case 3:
            alpha = 0.025
        case 4:
            alpha = 0.05
        case 5:
            alpha = 0.1
        case 6:
            alpha = 0.2
        case _:
            alpha = 0.05

    tcrit = stats.t.ppf(1-alpha/2, n-p)
    err = tcrit * bse
    normalCond = np.abs(np.sign(beta + err) + np.sign(beta - err))

    color = np.where(beta > 0, "green", "red")
    color = np.where(normalCond < 2, "lightgray", color)

    # Create the plot
    fig1 = go.Figure()

    fig1.add_trace(go.Scatter(
        x=beta,
        y=pred,
        mode="markers",
        marker=dict(
            color=color,
            size=10,
            line=dict(
                color="black",
                width=1
            )
        ),
        name="Coefficient",
        error_x=dict(array=err, color="black"),
        hoverinfo="y",
    ))

    fig1.update_layout(
        template="simple_white",
        xaxis_title="Coefficient Size",
        yaxis=dict(
            title_font=dict(size=14, color="black"),
            tickvals=pred,
            ticktext = [p.split("_", 1)[-1] for p in pred],
            tickfont=dict(family="Gotham"),
            #tickangle=-45,
        ),
        xaxis=dict(
            title_font=dict(size=14, color="black", family="Gotham")
        ),
        legend=dict(
            x=1,
            y=1,
            traceorder="normal",
            orientation="v"
        ),
        margin=dict(l=10, r=10, t=0, b=0),
        hovermode="closest",
        height=375,
    )

    fig1.update_xaxes(showgrid=True, gridwidth=1, gridcolor="LightGray")
    return fig1


@callback(
    Output("confidence", "children"),
    Input("alpha-slider", "value"),
)
def update_confidence(alpha):
    match alpha:
        case 1:
            alpha = "99.9%"
        case 2:
            alpha = "99.0%"
        case 3:
            alpha = "97.5%"
        case 4:
            alpha = "95.0%"
        case 5:
            alpha = "90.0%"
        case 6:
            alpha = "80.0%"
        case _:
            alpha = "95.0%"
    return alpha


@callback(
    Output("coeff-caption", "children"),
    Input("coeff-plot", "hoverData"),
    Input("trans-selection", "value"),
    State('session', "data"),
)
def update_coeff_caption(hoverData, trans, data):

    if hoverData is None:
        session_id = data.get("session_id", None)
        model = query_model(session_id)
        if model is None:
            return "No Data Available"

        r2 = model["r2"]
        ar2 = model["ar2"]
        aic = model["aic"]
        bic = model["bic"]

        content = html.Div([
            html.Span(f"R2: ", style={"color": GRAY}),
            html.Span('{:20,.1f}%'.format(100 * r2)),
            html.Span(f", Adj. R2: ", style={"color": GRAY}),
            html.Span('{:20,.1f}%'.format(100 * ar2)),
            html.Span(f", AIC: ", style={"color": GRAY}),
            html.Span('{:20,.1f}'.format(aic)),
            html.Span(f", BIC: ", style={"color": GRAY}),
            html.Span('{:20,.1f}'.format(bic)),
        ])

        return content

    beta = hoverData["points"][0]["x"]
    var = hoverData["points"][0]["y"]
    color = hoverData["points"][0]["marker.color"]

    color = FORESTGREEN if color == "green" else color
    color = RED if color == "red" else color
    color = GRAY if color == "lightgray" else color

    var_split = var.split("_", 1)
    perc = 100 * (np.exp(beta) - 1)

    if var in ["Intercept", "const"]:
        content = html.Div([
            html.Span(f"Expected pay when all predictors are base/zero is "),
            html.Span(f"{'${:20,.2f}'.format(abs(beta))}", style={"color": color}),
            html.Span(f"."),
        ])

        return content

    elif len(var_split) == 2:
        p = var_split[-1]
        cat = var_split[0]

        if trans == "log":
            change = "lower" if perc < 0 else "higher"
            num = '{:20,.1f}%'.format(abs(perc))
        else:
            change = "lower" if beta < 0 else "higher"
            num = '${:20,.2f}'.format(abs(beta))

        content = html.Div([
            html.Span(f"For "),
            html.Span(f"{p}", style={"color": color}),
            html.Span(f" employees, expected pay is "),
            html.Span(f"{num} {change}", style={"color": color}),
            html.Span(f" than base."),
        ])

        return content

    else:
        p = var
        cat = None

        if trans == "log":
            change = "decreases" if perc < 0 else "increases"
            num = '{:20,.1f}%'.format(abs(perc))
        else:
            change = "decreases" if beta < 0 else "increases"
            num = '${:20,.2f}'.format(abs(beta))


        content = html.Div([
            html.Span(f"For every one unit increase in "),
            html.Span(f"{p}", style={"color": color}),
            html.Span(f", expected pay "),
            html.Span(f"{change}", style={"color": color}),
            html.Span(f" by "),
            html.Span(f"{num}", style={"color": color}),
            html.Span(f"."),
        ])
        return content


@callback(
    Output("res-plot-title", "children"),
    Output("res-plot", "figure"),
    Input("res-type", "value"),
    Input("session", "data"),
)
def plot_res(res_type, data):
    session_id = data.get("session_id", None)
    model = query_model(session_id)

    if model is None:
        return "Residual Plot", EMPTYFIG
    else:
        n = model["n"]
        fit = model["fit"]
        res = model["res"]
        actual = fit + res
        stu = model["stu"]
        cookd = model["cookd"]


    color = np.where(res < 0, "#d40a1c", "#00173b")

    match res_type:
        case 1: # Actual vs. Expected Pay
            title = "Actual vs. Expected Pay"
            x = fit
            y = actual
            xtitle = "Expected Pay"
            ytitle = "Actual Pay"

        case 2: # Residuals
            title = "Residuals"
            x = fit
            y = res
            xtitle = "Expected Pay"
            ytitle = "Residuals"

        case 3: # Standardized Residuals
            title = "Standardized Residuals"
            x = fit
            y = res / np.std(res)
            xtitle = "Expected Pay"
            ytitle = "Standardized Residuals"

        case 4: # Studentized Residuals
            title = "Studentized Residuals"
            x = fit
            y = stu
            xtitle = "Expected Pay"
            ytitle = "Studentized Residuals"

        case 5: # Normal Q-Q of Residuals
            title = "Normal Q-Q of Residuals"
            x = stats.norm.ppf((np.arange(n) + 1) / (n + 1))
            x = x[res.argsort().argsort()]
            y = res
            xtitle = "Theoretical Normal Quantiles"
            ytitle = "Actual Residual Quantiles"

        case 6: # Cook's Distance
            title = "Cook's Distance"
            x = fit
            y = cookd
            xtitle = "Expected Pay"
            ytitle = "Cook's Distance"

        case _:
            title = "Residuals"
            x = fit
            y = res
            xtitle = "Expected Pay"
            ytitle = "Residuals"

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=x,
        y=y,
        text=["$" + '{:20,.2f}'.format(a).strip() for a in actual],
        mode="markers",
        marker=dict(size=0, color=alpha(color, 0.4)),
        hoverinfo="text",
    ))
    fig2.update_layout(
        template="simple_white",
        xaxis_title=xtitle,
        yaxis_title=ytitle,
        xaxis_title_font=dict(size=14, family="Gotham"),
        yaxis_title_font=dict(size=14, family="Gotham"),
        margin=dict(l=10, r=10, t=0, b=0),
        height=430,
    )

    return title, fig2


@callback(
    Output("info-body", "children"),
    Input("res-plot", "relayoutData"),
    Input("res-plot", "figure"),
    State("session", "data")
)
def handle_resize(relayoutData, fig, data):
    default = html.P("No Data Available")
    session_id = data.get("session_id", None)
    df, _ = query_data(session_id)

    if df is None:
        return default

    figdata = fig["data"]

    if relayoutData is None:
        i = np.array(df.index)
    elif not figdata:
        i = np.array(df.index)
    else:
        x = np.array(figdata[0]["x"])
        y = np.array(figdata[0]["y"])

        if "xaxis.autorange" in relayoutData:
            xi = np.array(df.index)

        else:
            try:
                x0 = relayoutData["xaxis.range[0]"]
                x1 = relayoutData["xaxis.range[1]"]
            except KeyError:
                xi = np.array(df.index)
            else:
                condition = (x0 <= x) & (x <= x1)
                xi = np.where(condition)


        if "yaxis.autorange" in relayoutData:
            yi = np.array(df.index)
        else:
            try:
                y0 = relayoutData["yaxis.range[0]"]
                y1 = relayoutData["yaxis.range[1]"]
            except KeyError:
                yi = np.array(df.index)
            else:
                condition = (y0 <= y) & (y <= y1)
                yi = np.where(condition)

        i = np.intersect1d(xi, yi)

    if i.size == 0:
        return default


    rows = df.iloc[i]
    box = dag.AgGrid(
        rowData=rows.to_dict("records"),
        columnDefs=[{"field": c} for c in rows.columns],
        columnSize="sizeToFit",
        #dashGridOptions={'pagination':True},
        style={"height": "270px"},
    )
    return box

@callback(
    Output("res-plot", "relayoutData"),
    Input("res-plot", "figure")
)
def reset_relayout_data(fig):
    return None

@callback(
    Output('coeff-plot', 'clickData'),
    Output('coeff-plot', 'hoverData'),
    Output('res-plot', 'clickData'),
    Output('res-plot', 'hoverData'),
    Input("quant-pred", "value"),
    Input("cat-pred", "value"),
    Input({"type": "dummy-dropdown", "index": ALL}, "value"),
    Input("trans-selection", "value"),
)
def reset_click_data(*args):
    return None, None, None, None

dash.register_page(
    __name__,
    path=PATH,
    title=NAME,
    name=NAME,
    order=6,
    layout=layout,
    default=page_layout(),
)
