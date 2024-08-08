

import dash
from dash import Dash, dcc, html, Input, Output, State, callback, ALL, ctx, no_update, dash_table
import dash_ag_grid as dag
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
from utils.cache import query_data, query_comparisons, query_model
from utils.config import BLUE, PINK, FORESTGREEN, GRAPHCONFIG, EMPTYFIG, BASICCOMPS, AXISBLANK, MARGINBLANK, alpha, RED, GRAY
import statsmodels.api as sm
import scipy.stats as stats
import plotly.graph_objects as go
import plotly.express as px
from dash.exceptions import PreventUpdate

NAME = "Remediation Modeling"
PATH = "/remediation-modeling"

layout = html.Div()

font_test = html.Div([
    html.P("The fat cat jumped over the white fence.", style={"font-family": "Gotham Thin"}),
    html.P("The fat cat jumped over the white fence.", style={"font-family": "Gotham ExtraLight"}),
    html.P("The fat cat jumped over the white fence.", style={"font-family": "Gotham Light"}),
    html.P("The fat cat jumped over the white fence.", style={"font-family": "Gotham"}),
    html.P("The fat cat jumped over the white fence.", style={"font-family": "Gotham Medium"}),
    html.P("The fat cat jumped over the white fence.", style={"font-family": "Gotham Bold"}),
    html.P("The fat cat jumped over the white fence.", style={"font-family": "Gotham Black"}),
    html.P("The fat cat jumped over the white fence.", style={"font-family": "Gotham Ultra"}),
])

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
    #disabled=True,
)

cat_pred = dcc.Dropdown(
    id="cat-pred",
    multi=True,
    placeholder="Select categorical predictor(s) of pay",
    persistence=True,
    persistence_type="session",
    #disabled=True,
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

res_plot = dcc.Graph(
    id="res-plot",
    config=GRAPHCONFIG,
    figure=EMPTYFIG,
    className="px-3 py-3",
)

info_box = html.Div([
    html.Div(id="info-hidden", hidden=True),
    html.H5(id="info-type", className="px-3 pt-3", children="Beta Estimate"),
    html.H4(id="info-label", className="px-3"),
    html.Div(id="info-body", className="px-3"),
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
    layout = html.Div(id="layout", children=[
        dbc.Container([
            html.P("This is page 5"),
            trans_selection,
            quant_pred,
            cat_pred,
            dummy_pred,
            model_btn,
            dbc.Row([
                dbc.Col(coeff_plot, width=6),
                dbc.Col(res_plot, width=6),
            ])
        ], className="my-4"),
    ])

    layout = dbc.Container(id="layout", fluid=True, style={"height": "100%", "padding": "0px"}, children=[
        dbc.Row(style={"height": "100%"}, children=[
            dbc.Col(id="sidebar", width=3, children=[
                sideinputs,
            ]),
            dbc.Col(width=9, style={"height": "100%"}, children=[
                dbc.Row(style={"height": "65%"}, children=[
                    dbc.Col(id="plot-main", width=6, children=[
                        html.Div(className="plot-frame", children=[
                            coeff_plot
                        ])
                    ]),
                    dbc.Col(id="plot-side", width=6, children=[
                        html.Div(className="plot-frame", children=[
                            res_plot
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
            #dbc.Col(id="plot-main", width=5, children=[
            #    html.Div(className="plot-frame", children=[
            #        #caption,
            #        #dumbbells,
            #        coeff_plot
            #    ])
            #]),
            #dbc.Col(id="plot-side", width=4, children=[
            #    html.Div(className="plot-frame", children=[
            #        #level,
            #        #metrics,
            #        #boxplots,
            #        res_plot
            #    ])
            #]),
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
    timestamp = data.get('timestamp', None)

    comparisons = query_comparisons(session_id, timestamp)

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
    timestamp = data.get('timestamp', None)

    comparisons = query_comparisons(session_id, timestamp)

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

        #print(dummies)

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
    #Output("coeff-plot", "figure"),
    #Output("res-plot", "figure"),
    Output("session", "data", allow_duplicate=True),
    Input("quant-pred", "value"),
    Input("cat-pred", "value"),
    Input({"type": "dummy-dropdown", "index": ALL}, "value"),
    Input("trans-selection", "value"),
    #Input("extra-options", "value"),
    State("session", "data"),
    prevent_initial_call=True,
)

def update_coeff_plot(quant_preds, cat_preds, dummy_lists, trans, data): #extra, data):

    print(f"dummy_lists: {dummy_lists}")
    session_id = data.get('session_id', None)
    timestamp = data.get('timestamp', None)

    df, _ = query_data(session_id)

    if df is None:
        raise PreventUpdate
        #return EMPTYFIG, EMPTYFIG

    comparisons = query_comparisons(session_id, timestamp)

    if comparisons is None:
        raise PreventUpdate
        #return EMPTYFIG, EMPTYFIG

    response = comparisons["response"]

    cat_dict = {}
    if cat_preds is not None:
        for i, c in enumerate(cat_preds):
            print(f"enumerate: {i} {c} / {len(dummy_lists)}")
            try:
                dummies = dummy_lists[i]
                if dummies:
                    cat_dict[c] = dummies
            except IndexError as err:
                print(f"Index Error while iterating through lists of dummy variables. Returning empty plots. Message: {err}")
                raise PreventUpdate
                #return EMPTYFIG, EMPTYFIG

    if not quant_preds and not cat_dict:
        raise PreventUpdate
        #return EMPTYFIG, EMPTYFIG


    y, X = preprocess(df, quant_preds, cat_dict, response, trans)
    model = query_model(session_id, timestamp, y, X)

    if model is None:
        data["valid_model"] = False
    else:
        data["valid_model"] = True

    return data

    #fig1, fig2 = plot_linear_model_coefficients(df, quant_preds, cat_dict, response, trans, extra)
    #return fig1, fig2


"""
@callback(
    Output("coeff-plot", "figure"),
    Output("res-plot", 'figure'),
    Input("model-btn", "n_clicks"),
    State("var-selection", 'value'),
    State("session", "data"),
    State("trans-selection", "value")
)
def update_coeff_plot(n_clicks, vars, data, trans):
    if vars is None:
        return EMPTYFIG, EMPTYFIG
    session_id = data.get('session_id', None)
    timestamp = data.get('timestamp', None)
    df, _ = query_data(session_id)
    comparisons = query_comparisons(session_id, timestamp)

    if df is None:
        return EMPTYFIG, EMPTYFIG

    if comparisons is None:
        return EMPTYFIG, EMPTYFIG

    fig1, fig2 = plot_linear_model_coefficients(df, vars, "pay", trans)
    return fig1, fig2
"""

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
    Input("session", "data"),
)
def plot_coeff(extra, data):
    session_id = data.get("session_id", None)
    timestamp = data.get("timestamp", None)

    model = query_model(session_id, timestamp)

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

    alpha = 0.05
    tcrit = stats.t.ppf(1-alpha/2, n-p)
    err = tcrit * bse
    normalCond = np.abs(np.sign(beta + err) + np.sign(beta - err))

    color = np.where(beta > 0, "green", "red")
    color = np.where(normalCond < 2, "gray", color)

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
    ))

    fig1.update_layout(
        title="Linear Model Coefficients with Error Bars",
        yaxis_title="Predictor",
        xaxis_title="Coefficient",
        yaxis=dict(
            title_font=dict(size=14, color="black"),
            tickvals=pred,
            ticktext = [p.split("_", 1)[-1] for p in pred],
            #tickangle=-45,
        ),
        xaxis=dict(
            title_font=dict(size=14, color="black")
        ),
        legend=dict(
            x=1,
            y=1,
            traceorder="normal",
            orientation="v"
        ),
        margin=dict(l=40, r=40, t=40, b=40)
    )

    fig1.update_xaxes(showgrid=True, gridwidth=1, gridcolor="LightGray")
    fig1.update_yaxes(showgrid=True, gridwidth=1, gridcolor="LightGray")

    return fig1


@callback(
    Output("res-plot", "figure"),
    Input("session", "data"),
)
def plot_res(data):
    session_id = data.get("session_id", None)
    timestamp = data.get("timestamp", None)

    model = query_model(session_id, timestamp)

    if model is None:
        return EMPTYFIG

    fit = model["fit"]
    res = model["res"]


    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=fit,
        y=res,
        mode="markers",
        marker=dict(size=0, color="rgba(219, 64, 82, 0.8)"),
        name="Residuals"
    ))

    fig2.update_layout(
        title="Residual Plot",
        xaxis_title="Fitted Values",
        yaxis_title="Residuals",
    )

    return fig2



def plot_linear_model_coefficients(df, quant_preds, cat_dict, y, trans, extra):
    """
    Create a plot of linear model coefficients with error bars for given predictors and target variable.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.
    predictors (list): List of column names to be used as predictors.
    y (str): The name of the target variable.

    Returns:
    go.Figure: A plotly Figure object with the coefficients and error bars.
    """
    X = df[quant_preds]
    for cat in cat_dict:
        df_dummies = pd.get_dummies(df[cat], prefix=cat).astype(int)
        dummies = cat_dict[cat]
        df_dummies = df_dummies[[f"{cat}_{d}" for d in dummies]]
        X = pd.concat([X, df_dummies], axis=1)

    X = X.dropna()
    X = sm.add_constant(X)

    y = df[y][X.index]
    if trans == "log":
        c = 0.001
        y = np.where(y > 0, np.log(y), c)

    # Fit the linear model
    model = sm.OLS(y, X).fit()

    # Extract coefficients and standard errors
    pred = model.params.index
    beta = model.params.values
    bse = model.bse
    fit = model.fittedvalues
    res = model.resid

    # remove intercept
    if 0 in extra:
        pred = pred[1:]
        beta = beta[1:]
        bse = bse[1:]

    alpha = 0.05
    zcrit = norm.ppf(1-alpha/2)
    err = zcrit * bse
    normalCond = np.abs(np.sign(beta + err) + np.sign(beta - err))

    color = np.where(beta > 0, "green", "red")
    color = np.where(normalCond < 2, "gray", color)

    # Create the plot
    fig1 = go.Figure()

    """
    for i, (p, b, e) in enumerate(zip(pred, beta, err)):
        fig1.add_shape(
            type="rect",
            x0=p,
            x1=p,
            y0=b-e,
            y1=b+e,
            line=dict(color="red", width=4),
            fillcolor="red",
            layer="below",
            name="Error Bar"
        )
    """


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
    ))


    fig1.update_layout(
        title="Linear Model Coefficients with Error Bars",
        yaxis_title="Predictor",
        xaxis_title="Coefficient",
        yaxis=dict(
            title_font=dict(size=14, color="black"),
            tickvals=pred,
            ticktext = [p.split("_", 1)[-1] for p in pred],
            #tickangle=-45,
        ),
        xaxis=dict(
            title_font=dict(size=14, color="black")
        ),
        legend=dict(
            x=1,
            y=1,
            traceorder="normal",
            orientation="v"
        ),
        margin=dict(l=40, r=40, t=40, b=40)
    )

    fig1.update_xaxes(showgrid=True, gridwidth=1, gridcolor="LightGray")
    fig1.update_yaxes(showgrid=True, gridwidth=1, gridcolor="LightGray")


    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=fit,
        y=res,
        mode="markers",
        marker=dict(size=0, color="rgba(219, 64, 82, 0.8)"),
        name="Residuals"
    ))

    fig2.update_layout(
        title="Residual Plot",
        xaxis_title="Fitted Values",
        yaxis_title="Residuals",
    )

    return fig1, fig2



@callback(
    Output("info-type", "children"),
    Output("info-hidden", "children"),
    Input('coeff-plot', 'hoverData'),
    Input('coeff-plot', 'clickData'),
    Input('res-plot', 'hoverData'),
    Input('res-plot', 'clickData'),
)
def handle_hover_data(coeffHover, coeffClick, resHover, resClick):
    isClick = coeffClick is not None or resClick is not None
    default_info = (no_update, "")

    def get_info(hover, click, value, key):
        if click is not None:
            return value, click["points"][0][key]
        elif isClick:
            raise PreventUpdate
        elif hover is not None:
            return value, hover["points"][0][key]
        else:
            return default_info

    trigger = next(iter(ctx.triggered_prop_ids))

    if trigger == "coeff-plot.hoverData":
        return get_info(coeffHover, None, "Beta Estimate", "y")

    if trigger == "coeff-plot.clickData":
        return get_info(coeffHover, coeffClick, "Beta Estimate", "y")

    if trigger == "res-plot.hoverData":
        return get_info(resHover, None, "Residual", "pointIndex")

    if trigger == "res-plot.clickData":
        return get_info(resHover, resClick, "Residual", "pointIndex")

    raise PreventUpdate



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


@callback(
    Output("info-label", "children"),
    Output("info-body", "children"),
    Input("info-type", "children"),
    Input("info-hidden", "children"),
    State("session", "data"),
)

def update_info_box(info_type, info_value, data):
    default_info = ("No Data Available", html.P(className="px-3", children="No Data Available"))

    if not info_value:
        return default_info


    session_id = data.get("session_id", None)
    timestamp = data.get("timestamp", None)

    model = query_model(session_id, timestamp)
    if model is None:
        return default_info

    if info_type == "Beta Estimate":

        """
        var_split = info_value.split("_", 1)
        if len(var_split) == 1:
            label = var_split[-1]
        elif len(var_split) == 2:
            label = f"{var_split[-1]} ({var_split[0].capitalize()})"
        else:
            label = info_value

        return label, "body test"
        """

        label, body = coeff_info(info_value, model)
        return label, body



    if info_type == "Residual":
        if not isinstance(info_value, int):
            return default_info

        df, _ = query_data(session_id)

        if df is None:
            return default_info

        """
        print(f"info_value i: {info_value}")
        try:
            row = df.iloc[info_value]
        except TypeError as err:
            print(f"Type Error while trying to locate row of residual. Returning 'No Data Available'. Message: {err}")
            return default_info

        return info_value, str(row)
        """

        label, body = res_info(info_value, df, model)
        return label, body

    raise PreventUpdate


def coeff_info(var, model):
    var_split = var.split("_", 1)
    if len(var_split) == 2:
        p = var_split[-1]
        cat = var_split[0]
    else:
        p = var
        cat = None

    n = model["n"]
    pn = model["p"]
    pred = model["pred"]
    beta = model["beta"]
    bse = model["bse"]

    i = pred.get_loc(var)
    b = beta[i]
    se = bse[i]



    alpha = 0.05
    tcrit = stats.t.ppf(1-alpha/2, n-pn)
    e = tcrit * se

    tstat = b/se
    print(-abs(tstat))
    p_val = stats.t.cdf(-abs(tstat), n-pn)
    print(p_val)

    coeff_stats = html.Div(id="metric-display", children=[
        html.H5(className="pt-2", children=[
            html.Div("T Stat:", className="metric-label"),
            html.Div('{:20,.4f}'.format(tstat), className="metric-value tabular"),
        ]),
        html.H5(className="pt-2", children=[
            html.Div("P Value:", className="metric-label"),
            html.Div('{:20,.4f}'.format(p_val), className="metric-value tabular"),
        ]),
        html.H5(className="pt-2", children=[
            html.Div("95% Confidence Interval:", className="metric-label"),
            html.Div('{:20,.2f}'.format(b-e) + ", " + '{:20,.2f}'.format(b+e), className="metric-value tabular"),
        ])
    ])

    content = html.Div(className="px-3", children=[
        dbc.Row([
            dbc.Col(width=6, children=[
                html.H5('β = {:20,.2f}'.format(b)),
            ]),
            dbc.Col(width=6, children=[
                coeff_stats
            ]),
        ]),
        dbc.Row([
            dbc.Col([
                html.P(f"For every one unit increase in {p}, predicted pay increases by " + '${:20,.2f}'.format(b) + ", all else equal.")
            ]),
        ]),

    ])

    label = f"{p}" if cat is None else f"{p} ({cat})"


    return label, content

def res_info(i, df, model):
    print(f"i: {i}")

    fit = model["fit"]
    res = model["res"]
    stu = model["stu"]
    lev = model["lev"]
    cookd = model["cookd"]

    std_res = res / np.std(res)

    kde = stats.gaussian_kde(std_res)
    x = np.linspace(np.floor(min(std_res)), np.ceil(max(std_res)), 100)
    y = kde.evaluate(x)

    xi = std_res[i]
    yi = kde.evaluate(xi)[0]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        mode="lines",
        fill="tozeroy",
        line=dict(color=alpha(RED, 0.1))
    ))

    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        mode="lines",
        line=dict(color=alpha(RED, 0.5), width=2)
    ))

    fig.add_trace(go.Scatter(
        x=[xi-0.25, xi+0.25, xi, xi, xi-0.25, xi+0.25],
        y=[0, 0, 0, max(y), max(y), max(y)],
        mode="lines",
        #marker=dict(color="red", size=15),
        line=dict(color="red", width=2),
        name="Highlighted Residual"
    ))

    fig.layout = go.Layout(yaxis=AXISBLANK, margin=MARGINBLANK)
    fig.update_layout(
        showlegend=False,
        template="simple_white",
        width=300,
        height=150,
        xaxis_title="Standardized Residual"
    )

    graph = dcc.Graph(figure=fig, config=GRAPHCONFIG)


    labcolor = "positive" if res[i] >= 0 else "negative"


    pay_caption = html.Div(id="metric-display", children=[
        html.H5(className="pt-2", children=[
            html.Div("Actual Pay:", className="metric-label"),
            html.Div('${:20,.2f}'.format(fit[i]+res[i]), className="metric-value tabular"),
        ]),
        html.H5(className="pt-2", children=[
            html.Div("Predicted Pay:", className="metric-label"),
            html.Div('${:20,.2f}'.format(fit[i]), className="metric-value tabular"),
        ]),
        html.H5(className="pt-2", children=[
            html.Div("Residual Pay:", className="metric-label"),
            html.Div('${:20,.2f}'.format(res[i]), className=f"metric-value tabular {labcolor}"),
        ])
    ])

    outlier_caption = html.Div(id="metric-display", children=[
        html.H5(className="pt-2", children=[
            html.Div("Studentized Residual:", className="metric-label"),
            html.Div('{:20,.4f}'.format(stu[i]), className="metric-value tabular"),
        ]),
        html.H5(className="pt-2", children=[
            html.Div("Leverage:", className="metric-label"),
            html.Div('{:20,.4f}'.format(lev[i]), className="metric-value tabular"),
        ]),
        html.H5(className="pt-2", children=[
            html.Div("Cook's Distance:", className="metric-label"),
            html.Div('{:20,.4f}'.format(cookd[i]), className="metric-value tabular"),
        ])
    ])

    row = df.iloc[[i]]

    row_table = dash_table.DataTable(
        data=df.iloc[[i]].to_dict("records"),
        #columns=[{"name": i, "id": i} for i in df.columns],
        #page_size=10,
        style_table={'width': "100%", "overflowX": "auto"},
        style_cell={"font-family": "Gotham Thin"},
        fixed_rows={'headers': True},
        #fixed_columns={'headers': True, 'data':2},
    )


    """
    row_table = dag.AgGrid(
        rowData=row.to_dict("records"),
        columnDefs=[{"field": c} for c in row.columns],
        columnSize="sizeToFit",
    )
    """

    content = html.Div(className="px-3", children=[
        dbc.Row([
            dbc.Col(width=4, children=[
                pay_caption,
            ]),
            dbc.Col(width=4, children=[
                graph,
            ]),
            dbc.Col(width=4, children=[
                outlier_caption,
            ]),
        ]),
        dbc.Row([
            dbc.Col(width=12, children=[
                row_table,
            ])
        ])
    ])

    return f"Employee #{i}", content


dash.register_page(
    __name__,
    path=PATH,
    title=NAME,
    name=NAME,
    order=6,
    layout=layout,
    default=page_layout(),
)
