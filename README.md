# BGG Recommandation
 
 ## Objective

 This project recommands similar boardgames from a selected game.

 ## Source

The games provide from the 2000 best rank BGG games.

BGG : BoardGameGeek is a website dedicated for boardgames.
https://boardgamegeek.com/


There are two sources from BGG :
* Scrapping to get game IDs
* API request to get information about games

### BoardGame Update
There is a screen inside the interface to update boardgames and their information.

Click on the button.
When the script is finished, a "Fini" message appears.

## Interface

The interface is created with Dash library.

https://dash.plotly.com/

It is composed of 2 screens :
* Update
* Recommandation (Home)


## Recommandation system

The recommandation system is based on Nearest Neighbors library from SciKit-Learn

```python
from sklearn.neighbors import NearestNeighbors
```

To find a "similar" game, the script compares the below elements :
* BGG rank
* Duration by player
* Year
* Min player age
* Number of awards
* BGG weight score (Game complexity)
* BGG rating
* Mechanisms
* Categories
* Domains

Then it determine the nearest games from the selected one.

### Handle player number

A user can select a player number. 

The script get the pool rating by player number.

It will keep only games which have been recommanded for this number of players (Recommanded + Best >= 50%).

If there is no poll, the script considers that this number of players is recommanded.


### Handle editions

In BGG, a second edition is considered as a game.

With the CosineSimilarity library, the script determines the similarity score based on 
* game name
* creators' name

If the similarity score is too high, the game is retrieved from the list.

### Handle the complexixity rate (weight)

A user can select a max value for weight.

This feature aims to recommand a game with less complexity.

For example, I like to play Tzolk'in, but I think this game is too complex for me.

I can find a similar game wich is less complex.
