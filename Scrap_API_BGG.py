import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import requests
import pandas as pd
from time import sleep
import re
from xml.etree import ElementTree


def recommanded_nb_player(dico_poll , min_play, max_play):
    '''
    Parameters
    ----------
    dico_poll : Dictionnary
        dictionnary of suggested players' poll.
    min_play : INT
        number of min players
    max_play : INT
        number of max players

    Returns
    -------
    list_play : LIST
        With the number of players which is recommanded as more than 50%
    '''
    list_play = []
    # For each nb_players
    for i in range(min_play,max_play+1):
  
      # we find the key of nb_player in key's list
      for k in dico_poll.keys():
        if int(re.findall('\d+',k)[0])==i:
          # calculate % recommandation
          total = sum(dico_poll[k].values())
          nb_recommanded = dico_poll[k]['Best'] + dico_poll[k]['Recommended']
          # if % recommandation >= 50%, then append list with nb_players (afet, we will make a get_dummies)
          if nb_recommanded / total >=0.5:
            list_play.append(i)
    return list_play

def update():
    dic_game={'id':[],
                'name':[],
                'link':[],
                'bgg_rank':[]
              }
    #%% SCRAP BGG
    #Scrap des 20 premières pages du site BGG
    for p in range(1,21):
      try:
        sleep(1)

        #scrappe la page p
        link = f"https://boardgamegeek.com/browse/boardgame/page/{p}"
        html = requests.get(link)
        soup =  BeautifulSoup(html.text,'html.parser')

        #Récupère la liste des jeux de la page
        games_names = soup.find_all('a',{'class':'primary'})


        #Récupère les ID des jeux de la page
        dic_game['id']+=[id['href'].split('/')[2] for id in games_names ]

        #Récupère les noms des jeux de la page
        dic_game['name'] += [name.text.strip() for name in games_names ]

        #link game detail
        dic_game['link'] += ['https://boardgamegeek.com' + link['href'] for link in games_names]

        # Récupère la liste des rangs BGG
        games_rank = soup.find_all('td',{'class' : 'collection_rank'})
        dic_game['bgg_rank'] += [int(re.findall('\d+',rank.text)[0]) for rank in games_rank]
      except:
        pass


    #%% API

    # add information into dic_game
    dic_game['suggested_player'] = []
    dic_game['min_player'] =[]
    dic_game['max_player'] =[]
    dic_game['max_playtime'] =[]
    dic_game['ratio_play'] =[]
    dic_game['age'] =[]
    dic_game['mecha_list'] =[]
    dic_game['family_list'] =[]
    dic_game['domain_list'] =[]
    dic_game['cat_list'] =[]
    dic_game['designer_list'] =[]
    dic_game['awards'] =[]
    dic_game['Image_link'] =[]
    dic_game['weight'] = []
    dic_game['geek_rating'] = []
    dic_game['year']=[]


    for nb_jeux in range(0,2001,100):

        id_game = ','.join(dic_game['id'][nb_jeux-100:nb_jeux])

        # get information from url
        url = f'https://boardgamegeek.com/xmlapi/boardgame/{id_game}?stats=1'
        session = requests.session()
        response = session.get(url)

        if response.status_code == 200 :

            # parse XML response
            tree = ElementTree.fromstring(response.content)

            for game in tree:
                # Min Player
                try :
                    min_player = int(game.find('./minplayers').text)
                    dic_game['min_player']+= [min_player]
                except:
                    dic_game['min_player']+=[np.nan]

                # Max Player
                try:
                    max_player = int(game.find('./maxplayers').text)
                    dic_game['max_player']+= [max_player]
                except:
                    dic_game['max_player']+=[np.nan]

                # Max playtime
                try:
                    max_playtime = int(game.find('./maxplaytime').text)
                    dic_game['max_playtime']+= [max_playtime]
                except:
                    dic_game['max_playtime']+=[np.nan]

                # ratio playtime by player
                try:
                    dic_game['ratio_play']+= [round(max_playtime / max_player,2)]
                except:
                    dic_game['ratio_play']+=[np.nan]

                # age
                try:
                    dic_game['age']+= [int(game.find('./age').text)]
                except:
                    dic_game['age']+=[np.nan]

                # Link Image
                try:
                    dic_game['Image_link']+= [game.find("./image").text]
                except:
                    dic_game['Image_link']+=[np.nan]

                # nb awards
                try:
                    dic_game['awards']+= [len(game.findall('./boardgamehonor'))]
                except:
                    dic_game['awards']+=[np.nan]

                # categorie
                try:
                    dic_game['cat_list']+= [[cat.text for cat in game.findall('./boardgamecategory')]]
                except:
                    dic_game['cat_list']+=[np.nan]

                # mechanics
                try:
                    dic_game['mecha_list']+= [[mecha.text for mecha in game.findall('./boardgamemechanic')]]
                except:
                    dic_game['mecha_list']+=[np.nan]

                # family
                try:
                    dic_game['family_list']+= [[family.text for family in game.findall('./boardgamefamily')]]
                except:
                    dic_game['family_list']+=[np.nan]

                # subdomain
                try:
                    dic_game['domain_list']+= [[domain.text for domain in game.findall('./boardgamesubdomain')]]
                except:
                    dic_game['domain_list']+=[np.nan]

                # designers
                try:
                    dic_game['designer_list']+= [[designer.text for designer in game.findall('./boardgamedesigner')]]
                except:
                    dic_game['designer_list']+=[np.nan]

                # complexity weight
                try :
                    dic_game['weight'] += [round(float(game.find('./statistics/ratings/averageweight').text),1)]
                except:
                    dic_game['weight'] += [np.nan]

                # Year
                try:
                    dic_game['year'] += [int(game.find('./yearpublished').text)]
                except:
                    dic_game['year'] += [np.nan]

                #geek_rating
                try :
                    dic_game['geek_rating'] += [round(float(game.find('./statistics/ratings/bayesaverage').text),1)]
                except:
                    dic_game['geek_rating'] += [np.nan]

                # dict with the result of poll about suggested players
                dict_poll = {}
                for results in tree[0].findall('./poll/[@name="suggested_numplayers"]/results'):
                  k = results.attrib['numplayers']
                  v = {r.attrib['value'] : int(r.attrib['numvotes']) for r in results.findall('./result')}
                  dict_poll[k]=v

                # create a list with all the players' number suggested to play
                dic_game['suggested_player'] +=[recommanded_nb_player(dict_poll,min_player,max_player)]

    #%% DATAFRAME
    #Creation d'un Data Frame
    df = pd.DataFrame(dic_game)

    #%% Value == 0  --> Mean

    df['min_player'] = df['min_player'].apply(lambda x : x if x>0 else int(df['min_player'].mean()))
    df['max_player'] = df['max_player'].apply(lambda x : x if x>0 else int(df['max_player'].mean()))
    df['max_playtime'] = df['max_playtime'].apply(lambda x : x if x>0 else int(df['max_playtime'].mean()))
    df['age'] = df['age'].apply(lambda x : x if x>0 else df['age'].mean())


    #%%
    #Export du dataframe en Excel

    df.to_csv('data/database_BGG.csv',index=0,sep =';')
