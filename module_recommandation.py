import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer


def game_domain_category(df, k, y,c, nb_players, id_selected_game):
    # TREATMENT SUGGESTED PLAYER
    # for empty list, fill with all numbers of player
    # and transform int in str before ','.join()
    df['suggested_player'] = df.apply(
        lambda row: row['suggested_player'] if len(row['suggested_player']) > 0 else [i for i in
                                                                                      range(row['min_player'],
                                                                                            row['max_player'] + 1)],
        axis=1)
    # si on a renseigné un nombre de joueurs pour jouer, on flag les jeux recommandés pour ce nombre de joueurs
    if nb_players is not None:
        df['player_flag'] = df['suggested_player'].apply(lambda mylist: 1 if nb_players in mylist else 0)
    # sinon on met 1 tout le temps
    else:
        df['player_flag'] = 1

    # TREATMENT DOMAIN (get dummies)
    df['domain_list_str'] = df['domain_list'].apply(lambda x: ','.join(x))
    df_domain = df['domain_list_str'].str.get_dummies(',')
    col_domain = df_domain.columns.to_list()
    df = pd.concat([df, df_domain], axis=1)
    df.drop(['domain_list_str'], axis=1, inplace=True)

    '''
    # TREATMENT CAT_LIST (get dummies)
    df['cat_list_str'] = df['cat_list'].apply(lambda x: ','.join(x))
    df_cat = df['cat_list_str'].str.get_dummies(',')
    col_cat = df_cat.columns.to_list()
    df = pd.concat([df, df_cat], axis=1)
    df.drop(['cat_list_str'], axis=1, inplace=True)
    df.fillna(0, inplace=True)
    '''
    # TREATMENT CAT_LIST (Cosine_similarity)
    # transform list into string
    df['cat_list_str'] = df['cat_list'].apply(lambda x: ','.join(x))

    # Vectorize & similarity calculation
    count_cat = CountVectorizer()
    count_matrix_cat = count_cat.fit_transform(df['cat_list_str'])
    cosine_sim_cat = cosine_similarity(count_matrix_cat, count_matrix_cat)

    # find the index of the selected game
    idx_selected_game = df[df['id'] == id_selected_game].index.to_list()[0]

    # filter on the index of the selected game
    score_series_cat = pd.Series(cosine_sim_cat[idx_selected_game]).sort_values(
        ascending=False)  # similarity scores in descending order*

    # join with the dataframe
    score_series_cat = score_series_cat.to_frame()
    score_series_cat.rename({0: 'match_cat'}, axis=1, inplace=True)
    df = df.join(score_series_cat)

    # TREATMENT MECHA with Cosine_similarity
    # transform list into string
    df['mecha_list_str'] = df['mecha_list'].apply(lambda x: ','.join(x))

    # Vectorize & similarity calculation
    count = CountVectorizer()
    count_matrix = count.fit_transform(df['mecha_list_str'])
    cosine_sim = cosine_similarity(count_matrix, count_matrix)

    # filter on the index of the selected game
    score_series = pd.Series(cosine_sim[idx_selected_game]).sort_values(
        ascending=False)  # similarity scores in descending order

    # join with the dataframe
    score_series = score_series.to_frame()
    score_series.rename({0: 'match_mecha'}, axis=1, inplace=True)
    df = df.join(score_series)

    ## TREATMENT COLLECTION GAME with Cosine_similarity
    # transform list into string
    df['designer_list_str'] = df['designer_list'].apply(lambda x: ','.join(x))

    # Join Name and designer
    df['game_designer'] = df.apply(lambda row: row['name'] + "," + row['designer_list_str'], axis=1)

    print(df['game_designer'].iloc[0:5])

    # Vectorize & similarity calculation
    count_col = CountVectorizer()
    count_matrix_col = count_col.fit_transform(df['game_designer'])
    cosine_sim_col = cosine_similarity(count_matrix_col, count_matrix_col)


    # find the index of the selected game
    idx_selected_game = df[df['id'] == id_selected_game].index.to_list()[0]

    # filter on the index of the selected game
    score_series_col = pd.Series(cosine_sim_col[idx_selected_game]).sort_values(
        ascending=False)  # similarity scores in descending order*

    # join with the dataframe
    score_series_col = score_series_col.to_frame()
    score_series_col.rename({0: 'match_col'}, axis=1, inplace=True)
    df = df.join(score_series_col)

    print(df['match_col'].describe())

    # RECOMMANDATION
    # split the selected game from the others
    df_game = df[df['id'] == id_selected_game]
    df_other_game = df[df['id'] != id_selected_game]

    # delete game where nb players is not selected
    df_other_game = df_other_game[df_other_game['player_flag'] == 1]
    print('other_game_shape', df_other_game.shape)

    # delete game in the same collection
    df_other_game = df_other_game[df_other_game['match_col'] < 0.7]
    print('other_game_shape', df_other_game.shape)

    if y is not None :
        # delete games which year is before ref year
        df_other_game = df_other_game[df_other_game['year']>= y]

    if c is not None :
        # delete games which complexity rate is higher than the selected value
        df_other_game = df_other_game[df_other_game['weight']<=c]

    col_X = ['bgg_rank',
             'ratio_play',
             'year',
             'age',
             'awards',
             'weight',
             'geek_rating',
             'match_mecha',
             'match_cat'] + col_domain

    # Definition X
    X = df_other_game[col_X]
    X_game = df_game[col_X]

    scaler = MinMaxScaler()

    # fit & transform train
    X_scaled = scaler.fit_transform(X)
    X_game_scaled = scaler.transform(X_game)

    # weight on some columns
    X_scaled[:, 5] = X_scaled[:, 5] * 7
    X_scaled[:, 7] = X_scaled[:, 7] * 10
    X_scaled[:, 8] = X_scaled[:, 8] * 3

    X_game_scaled[:, 5] = X_game_scaled[:, 5] * 7
    X_game_scaled[:, 7] = X_game_scaled[:, 7] * 10
    X_game_scaled[:, 8] = X_game_scaled[:, 8] * 3

    # fit the model
    modelKNN = NearestNeighbors(n_neighbors=k).fit(X_scaled)

    dist, indice = modelKNN.kneighbors(X_game_scaled)

    df_result = df_other_game.iloc[indice[0], :]

    print(df_result[['name','match_col']])

    return df_result
