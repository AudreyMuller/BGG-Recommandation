import dash
from dash import html
import dash_bootstrap_components as dbc

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SIMPLEX, dbc.icons.BOOTSTRAP], use_pages=True)


app.layout = dbc.Container([
    dbc.Row([
        html.H1('Recommandation de jeux de société', style={'color': '#d9230f'}),
    ]),


    # NAVBAR
    dbc.Row([
        dbc.Nav([dbc.NavLink([html.Div(page["name"], className="ms-2"), ],
                             href=page["path"],
                             active="exact",
                             id={'type': 'nav_link',
                                 'index': page["name"], })
                 for page in dash.page_registry.values()],
            vertical=False,
            pills=True,
            className="bg-light",)
    ]),

    # PAGES
    dbc.Row([
        dash.page_container
    ])

])

if __name__ == '__main__':
    app.run_server(debug=False, port=8001, use_reloader=False)
