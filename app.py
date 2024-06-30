
import dash
from dash import html, dcc, Output, Input, State, callback, callback_context
import dash_bootstrap_components as dbc

# initialize dash app
external_stylesheets = [
    dbc.themes.BOOTSTRAP,
    "/assets/styles.css"
]

app = dash.Dash(__name__, use_pages=True, external_stylesheets=external_stylesheets)
app.config.suppress_callback_exceptions = True

tabs = dbc.Nav([
    dbc.NavItem(dbc.NavLink(page["name"], href=page["path"], className="nav-tabs border border-3 border-success", active="exact"))
    for page in dash.page_registry.values()
    if page["module"] != "pages.not_found404"
], pills=True, justified=True, id="navbar", className="ml-auto mx-5 my-2")

header = html.Div(className="header-container", children=[
    dbc.Navbar([
        # App Title
        html.A(dbc.Row([
            dbc.Col(dbc.NavLink(html.Img(src="/assets/images/logo.png", height="30px"), href="https://www.kornferry.com/")),
            dbc.Col(dbc.NavbarBrand("Pay Equity Demo", className="ml-2")),
        ], align="center"), href="#"),
    ], color="light", dark=False)
])


footer = html.Footer([
    html.P("© 2024 Korn Ferry", className="col-md-4 mb-0 text-muted"),
    dbc.NavLink(html.Img(src="/assets/images/github.png", height="30px"),
                href="https://github.com/KF-camccaffrey/KF-Dash",
                className="col-md-4 d-flex align-items-center justify-content-center mb-3 pt-3 link-dark text-decoration-none"),
    dbc.ListGroup(className="nav col-md-4 mb-0 justify-content-end", horizontal=True, children=[
        dbc.NavLink("Home", href="", className="nav-link px-2 text-muted"),
        dbc.NavLink("Features", href="", className="nav-link px-2 text-muted"),
        dbc.NavLink("Pricing", href="", className="nav-link px-2 text-muted"),
        dbc.NavLink("About", href="", className="nav-link px-2 text-muted")
    ])
], className="fixed-bottom d-flex flex-wrap justify-content-between align-items-center px-4 border-top")

app.layout = dbc.Container([
    dcc.Location(id="url", refresh=False),
    header,
    tabs,
    dash.page_container,
    footer
], fluid=True)


if __name__ == "__main__":
    app.run_server(debug=True)
