import dash
import dash_bootstrap_components as dbc
from dash import html, callback
from dash import dcc
from dash.dependencies import Output, Input, ALL, State, MATCH, ALLSMALLER
import dash_bootstrap_components as dbc
import Scrap_API_BGG

dash.register_page(__name__, name="Mise à jour")

layout = dbc.Container([
    html.P(''),
    dbc.Button('Launch', id='update_btn',className='primary'),
    html.P(''),
    html.Div([], id='update_status')
])


@callback(
    Output('update_status', 'children'),
    Input('update_btn', 'n_clicks'),
    prevent_initial_call=True,
)
def add_new_kw(add_clicks):
    Scrap_API_BGG.update()
    return [html.P('Fini')]
