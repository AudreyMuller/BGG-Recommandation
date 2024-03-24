import dash
from dash import html, callback
from dash import dcc
from dash.dependencies import Output, Input, State
import dash_bootstrap_components as dbc
import pandas as pd
import module_recommandation as reco
import module_mise_en_page as mep

# read CSV
df = pd.read_csv('data/database_BGG.csv', sep=';')

df['suggested_player'] = df['suggested_player'].apply(eval)
df['mecha_list'] = df['mecha_list'].apply(eval)
df['family_list'] = df['family_list'].apply(eval)
df['domain_list'] = df['domain_list'].apply(eval)
df['cat_list'] = df['cat_list'].apply(eval)
df['designer_list'] = df['designer_list'].apply(eval)

# recalculate ratio_play
df['ratio_play'] = df['max_playtime'] / df['max_player']

# create a dico for the dropdown
dict_dropdown = df[['id', 'name']].set_index('id').to_dict()['name']

# Année minimum
min_year = df[['year']].min()
# Année maximum
max_year = df[['year']].max()

print(min_year)
print(max_year)

# ---------------------------------------------------------------------
dash.register_page(__name__, path='/', name="Home")

layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.P(''),
            dbc.Row([
                html.H6("Choisissez le nom d'un jeu (en anglais)"),
                dcc.Dropdown(dict_dropdown, multi=False, id='game_id')
            ]),
            html.P(''),
            dbc.Row([
                html.H6("Choisissez un nombre de joueurs"),
                dcc.Input(type='number', id='player_id')
            ]),
            html.P(''),
            dbc.Row([
                html.H6("Choisissez le nombre de jeux à recommander"),
                dcc.Input(value=5, type='number', id='k_id', required=True)
            ]),
            html.P(''),
            dbc.Row([
                html.H6("Choisissez l'année minimum'"),
                dcc.Input(type='number', id='year_id', required=False, min=min_year, max=max_year)
            ]),
            dbc.Row([
                html.P(''),
                dbc.Button('Launch', id='launch_btn', className='primary'),
                html.P(''),
            ]),
        ], width=3),

        #
        dbc.Col([
            html.Div([], id='results'),
        ])
    ]),
])


# call back & def to add a keywords
@callback(
    Output('results', 'children'),
    Input('launch_btn', 'n_clicks'),
    State('game_id', 'value'),
    State('player_id', 'value'),
    State('k_id', 'value'),
    State('year_id', 'value'),
    prevent_initial_call=True,
)
def recommandation(add_clicks, selected_game, selected_player, k, y):
    print(selected_game)
    print(selected_player)
    print(y)
    selected_game = int(selected_game)
    result = reco.game_domain_category(df, k, y, selected_player, selected_game)
    result_str = ' - '.join(result['name'].to_list())

    df_selected_game = df[df['id'] == selected_game]
    df_concat = pd.concat([df_selected_game, result])
    print(df_concat.to_markdown())
    print(df_concat.iloc[0])
    result_div = [mep.mise_en_page_un_jeu(df_concat.iloc[i]) for i in range(df_concat.shape[0])]

    return result_div
