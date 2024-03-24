import dash
import dash_bootstrap_components as dbc
from dash import html
from dash import dcc
import dash_bootstrap_components as dbc



def mise_en_page_un_jeu(df):
    return dbc.Row([
                dbc.Col([
                    html.Img(src=df['Image_link'], height='300px', width='200px')
                    ], width=3),
                dbc.Col([
                    html.H4(df['name'], style={'color': '#D9230F', 'font-weight': 'bold'}),
                    dbc.NavLink("Lien vers BGG", href=df['link'], external_link=True),
                    html.P([html.Span("BGG Rank : ", style={'color': '#D9230F', 'font-weight': 'bold'}),
                            df['bgg_rank']]),
                    html.P([html.Span("Complexité : ", style={'color': '#D9230F', 'font-weight': 'bold'}),
                            df['weight']]),
                    html.P([html.Span("Age : ", style={'color': '#D9230F', 'font-weight': 'bold'}),
                            df['age']]),
                    html.P([html.Span("Min Player : ", style={'color': '#D9230F', 'font-weight': 'bold'}),
                            df['min_player']]),
                    html.P([html.Span("Max Player: ", style={'color': '#D9230F', 'font-weight': 'bold'}),
                            df['max_player']]),
                    html.P([html.Span("Durée : ", style={'color': '#D9230F', 'font-weight': 'bold'}),
                            df['max_playtime']]),
                    html.P([html.Span("Année : ", style={'color': '#D9230F', 'font-weight': 'bold'}),
                            df['year']]),
                    html.P([html.Span("Note : ", style={'color': '#D9230F', 'font-weight': 'bold'}),
                           df['geek_rating']]),
                    html.Hr()
                    ], width=3),
                dbc.Col([
                    html.P([html.Span("Créateurs : ", style={'color': '#D9230F', 'font-weight': 'bold'}),
                            ' - '.join(df['designer_list'])]),
                    html.P([html.Span("Mécanismes : ", style={'color': '#D9230F', 'font-weight': 'bold'}),
                            ' - '.join(df['mecha_list'])]),
                    html.P([html.Span("Domaines : ", style={'color': '#D9230F', 'font-weight': 'bold'}),
                            ' - '.join(df['domain_list'])]),
                    html.P([html.Span("Catégories : ", style={'color': '#D9230F', 'font-weight': 'bold'}),
                            ' - '.join(df['cat_list'])]),
                    ]),
                ])# end Row # end DIV children

