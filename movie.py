import numpy as np
import pandas as pd
import streamlit as st
import joblib
import pickle as pk
import difflib
from PIL import Image
import base64
import requests
from itertools import cycle


def fetch_poster(movie_id):
    url = "https://api.themoviedb.org/3/find/{}?api_key=6948ed89118150000547f12dca283524&language=en-US&external_source=imdb_id".format(
        movie_id)
    data = requests.get(url)
    data = data.json()
    # print(data)
    try:
        poster_path = data['movie_results'][0]['poster_path']
        full_path = "https://image.tmdb.org/t/p/w500/" + poster_path
        return full_path
    except:
        img = Image.open("./assets/images/What2watch.png")
        img = img.resize((500, 750))
        return img


profit_id_list = ['tt0499549', 'tt2488496', 'tt0120338', 'tt0369610', 'tt2820852', 'tt0848228', 'tt1201607',
                  'tt2395427', 'tt1323045', 'tt0113957', 'tt2293640', 'tt0167260', 'tt1300854', 'tt1399103', 'tt1074638']
profit_list = ['Avatar',
               'Star Wars: The Force Awakens',
               'Titanic',
               'Jurassic World',
               'Furious 7',
               'The Avengers',
               'Harry Potter and the Deathly Hallows: Part 2',
               'Avengers: Age of Ultron',
               'Frozen',
               'The Net',
               'Minions',
               'The Lord of the Rings: The Return of the King',
               'Iron Man 3',
               'Transformers: Dark of the Moon',
               'Skyfall']
popularity_list = ['Jurassic World',
                   'Mad Max: Fury Road',
                   'Interstellar',
                   'Guardians of the Galaxy',
                   'Insurgent',
                   'Captain America: The Winter Soldier',
                   'Star Wars',
                   'John Wick',
                   'Star Wars: The Force Awakens',
                   'The Hunger Games: Mockingjay - Part 1',
                   'The Hobbit: The Battle of the Five Armies',
                   'Avatar',
                   'Inception',
                   'Furious 7',
                   'The Revenant']
popularity_id_list = ['tt0369610',
                      'tt1392190',
                      'tt0816692',
                      'tt2015381',
                      'tt2908446',
                      'tt1843866',
                      'tt0076759',
                      'tt2911666',
                      'tt2488496',
                      'tt1951265',
                      'tt2310332',
                      'tt0499549',
                      'tt1375666',
                      'tt2820852',
                      'tt1663202']
vote_list = ['Inception',
             'The Avengers',
             'Avatar',
             'The Dark Knight',
             'Django Unchained',
             'The Hunger Games',
             'Iron Man 3',
             'The Dark Knight Rises',
             'Interstellar',
             'The Hobbit: An Unexpected Journey',
             'The Matrix',
             'Iron Man',
             'Mad Max: Fury Road',
             'Skyfall',
             'The Lord of the Rings: The Fellowship of the Ring']
vote_id_list = ['tt1375666',
                'tt0848228',
                'tt0499549',
                'tt0468569',
                'tt1853728',
                'tt1392170',
                'tt1300854',
                'tt1345836',
                'tt0816692',
                'tt0903624',
                'tt0133093',
                'tt0371746',
                'tt1392190',
                'tt1074638',
                'tt0120737']
rate_id_list = ['tt0111161',
                'tt0468569',
                'tt0068646',
                'tt0137523',
                'tt0110912',
                'tt0109830',
                'tt0816692',
                'tt1375666',
                'tt0167260',
                'tt2015381',
                'tt0080684',
                'tt2096673',
                'tt0133093',
                'tt2084970',
                'tt0076759']
rate_list = ['The Shawshank Redemption',
             'The Dark Knight',
             'The Godfather',
             'Fight Club',
             'Pulp Fiction',
             'Forrest Gump',
             'Interstellar',
             'Inception',
             'The Lord of the Rings: The Return of the King',
             'Guardians of the Galaxy',
             'The Empire Strikes Back',
             'Inside Out',
             'The Matrix',
             'The Imitation Game',
             'Star Wars']

director_list = ['DavidCronenberg',
                 'RobReiner',
                 'JohnCarpenter',
                 'WesCraven',
                 'BarryLevinson',
                 'BrianDePalma',
                 'TimBurton',
                 'JoelSchumacher',
                 'StevenSoderbergh',
                 'RonHoward',
                 'RidleyScott',
                 'StevenSpielberg',
                 'MartinScorsese',
                 'ClintEastwood',
                 'WoodyAllen']

actor_list = ['RobertDeNiro',
              'NicolasCage',
              'SamuelL.Jackson',
              'BruceWillis',
              'JohnnyDepp',
              'JohnCusack',
              'TomHanks',
              'MichaelCaine',
              'RobinWilliams',
              'SylvesterStallone',
              'ClintEastwood',
              'MorganFreeman',
              'SusanSarandon',
              'EddieMurphy',
              'HarrisonFord',
              'MerylStreep']

runtime_id = ['tt2044056',
              'tt0936501',
              'tt0185906',
              'tt0090015',
              'tt0088583',
              'tt0795176',
              'tt0374463',
              'tt0472027',
              'tt2948840',
              'tt0995832',
              'tt1453159',
              'tt0207275',
              'tt2396421',
              'tt0296310',
              'tt1878805']
runtime_list = ['The Story of Film: An Odyssey',
                'Taken',
                'Band of Brothers',
                'Shoah',
                'North and South, Book I',
                'Planet Earth',
                'The Pacific',
                'John Adams',
                'Life',
                'Generation Kill',
                'The Pillars of the Earth',
                'The 10th Kingdom',
                'Crystal Lake Memories: The Complete History of Friday the 13th',
                'The Blue Planet',
                'World Without End']


@st.cache(allow_output_mutation=True)
def load_models():

    lower_title_list = {}
    title_list = joblib.load(open('title_list.pkl', 'rb'))
    similarity_score = joblib.load(open('similarity2.pkl', 'rb'))

    final_df = joblib.load(open('movie_data', 'rb'))

    for i in range(len(title_list)):
        lower_title_list[title_list[i].lower()] = i

    return title_list, final_df, similarity_score, lower_title_list


title_list, final_df, similarity_score, lower_title_list = load_models()


@st.cache(allow_output_mutation=True)
def find_id(l):
    id_list = []
    for i in l:
        id_list.append(
            final_df[final_df['original_title'] == i].imdb_id.tolist()[0])

    return id_list


def recommend(title):
    # find closest title from title list
    movie_list = []
    title = title.lower()
    title = difflib.get_close_matches(
        title, list(lower_title_list.keys()), cutoff=0.4)[0]

    idx = final_df[final_df['original_title'] ==
                   title_list[lower_title_list[title]]].index[0]

    # idx_score = list(enumerate(similarity_score[idx]))
    # sort on basis of 2nd parameter which is similarity score
    req_movies = similarity_score[idx]
    cnt = 0

    for i in req_movies:
        movie_list.append(final_df['original_title'][i[0]])
        cnt += 1
        if(cnt == 25):
            break
    return movie_list


def show(col, title):
    cols = cycle(st.columns(5))
    l = []
    for i in range(15):
        l.append([col[i], title[i]])

    # images = []
    # for i in col:
    #     response = requests.get(
    #         f"https://imdb-api.com/en/API/Posters/{key}/{i}".format(key, i))
    #     json = response.json()
    #     if(len(json['posters']) > 0):
    #         link = json['posters'][0]['link']
    #         images.append(link)
    # for idx, image in enumerate(images):
    #     #st.header(title[idx % 5])
    #     next(cols).image(image, width=150)
    for i in range(3):
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            n = i*5+0
            url = fetch_poster(l[n][0])
            st.image(url)
            st.write(
                "[{}](https://www.imdb.com/title/{}/)\n".format(l[n][1], l[n][0]))
        with col2:
            n = i*5+1
            url = fetch_poster(l[n][0])
            st.image(url)
            st.write(
                "[{}](https://www.imdb.com/title/{}/)\n".format(l[n][1], l[n][0]))
        with col3:
            n = i*5+2
            url = fetch_poster(l[n][0])
            st.image(url)
            st.write(
                "[{}](https://www.imdb.com/title/{}/)\n".format(l[n][1], l[n][0]))
        with col4:
            n = i*5+3
            url = fetch_poster(l[n][0])
            st.image(url)
            st.write(
                "[{}](https://www.imdb.com/title/{}/)\n".format(l[n][1], l[n][0]))
        with col5:
            n = i*5+4
            url = fetch_poster(l[n][0])
            st.image(url)
            st.write(
                "[{}](https://www.imdb.com/title/{}/)\n".format(l[n][1], l[n][0]))


title_list, final_df, similarity_score, lower_title_list = load_models()


def main():

    st.sidebar.title("Developer's Contact")
    st.sidebar.markdown('[![Harsh-Dhamecha]'
                        '(https://img.shields.io/badge/Author-Abhinav%20Singh-brightgreen)]'
                        '(https://github.com/Abhisin90)')
    img = Image.open("./assets/images/What2watch.png")
    st.sidebar.image(
        img, use_column_width=True)

    st.sidebar.title("Facts Probably U Don't Know")
    select = st.sidebar.selectbox(
        'The Analyzer', ['Home', 'Directors with most Number Of movies', 'Actors With Most Number Of Movies', 'Most Used Words in The Titles', 'Top Rated Movies', 'Highest Grossers', 'Most Popular Movies', 'Movies With longest Runtime'])

    if(select == 'Home'):
        st.markdown("# WHAT 2 WATCH")
        # img = Image.open("D:\PROJECTS\home.jpg")
        # st.image(img, width=1120)
        try:
            movie_input = st.text_input("Enter Movie Name")
            if(st.button("Show Recommendations")):
                l = recommend(movie_input)

                # for i in l:
                #     st.subheader(i)
                show(find_id(l), l)
        except:
            st.markdown("OOOPS NO CLOSE MATCHES FOUND !!!!!")

    elif(select == 'Most Popular Movies'):
        st.header("15 Most Crowd Pleasing Movies")
        st.markdown(
            "Rankings Based On Popularity is One Of The Easiest Task To Do. [**Jurassic World**](https://en.wikipedia.org/wiki/Jurassic_World) is The most Popular Movie according to IMDB which is the most Popular movie Information site.")

        img = Image.open('./assets/images/jurasic.jpg')
        st.image(img)
        st.markdown(
            "Basically Popularity Of a movie is given by the number of page views of all the pages related to that movie")
        img2 = Image.open('./assets/images/top_15_popularity.png')
        st.image(img2)
        show(popularity_id_list, popularity_list)
    elif(select == 'Highest Grossers'):
        st.header("15 Highest Grossing Movies")
        st.markdown(
            "Films generate income from several revenue streams, including theatrical exhibition, home video, television broadcast rights, and merchandising. However, theatrical box office earnings are the primary metric for trade publications in assessing the success of a film, mostly because of the availability of the data compared to sales figures for home video and broadcast rights, but also because of historical practice.")

        img = Image.open('./assets/images/avatar.jpg')
        st.image(img)
        st.markdown(
            "James Cameron's [Avatar](https://en.wikipedia.org/wiki/Avatar_(2009_film)) reclaimed the highest grossing movie of all time with 2.8 Billon dolars.  ")
        img2 = Image.open('./assets/images/top_15_profit.png')

        st.image(img2)
        show(profit_id_list, profit_list)
    elif(select == 'Top Rated Movies'):
        st.header("15 Movies ranked by IMDB Rating")
        st.markdown(
            "IMDB uses the weighted rating formula to construct the Top movie chart by Rating. Audience rating is the closest thing to a definitive ranking.")
        img = Image.open('./assets/images/shawshank.jpg')
        st.image(img)
        st.markdown("[The Shawshank Redemption](https://en.wikipedia.org/wiki/The_Shawshank_Redemption) holds the Number 1 spot in the Top-250 English Movies listed by IMBd. It is one among the best movies ever made in World Cinema and applauded by many film critics.")
        img2 = Image.open('./assets/images/top_15_wr.png')

        st.image(img2)
        show(rate_id_list, rate_list)
    elif(select == 'Directors with most Number Of movies'):
        st.header("The Director's List ")
        st.markdown("[Woody Allen]() is one of the Directors with most number of films.He is a writer and director who bares his emotions and neuroses on screen. It's for this very reason that so many of his films feel so personal. ")
        img = Image.open('./assets/images/woody allen.jpg')
        st.image(img)
        img2 = Image.open('./assets/images/directors.jpg')

        st.subheader("The Top 15 Directors :")
        director_list.reverse()
        j = 1
        for i in director_list:
            st.markdown("**{}. {}**".format(j, [i][0]))
            j += 1
        st.image(img2, width=960)

    elif(select == 'Actors With Most Number Of Movies'):
        st.header("Most Frequent Actors :")
        st.markdown(
            "And The award For Most commonly starring actor goes to [RobertDeNiro](https://en.wikipedia.org/wiki/Robert_De_Niro). If you see 'quality' as a measure of the artistic merits of a movie, then you may chime more with the views of film critics, whereas if you link quality to audience enjoyment then the IMDb audience scores may make more sense to you.")
        img = Image.open("./assets/images/actor.jpg")
        img2 = Image.open("./assets/images/actors.jpg")
        st.image(img)

        st.markdown("Other Top Frequent Actors: ")

        j = 1
        for i in actor_list:
            st.markdown("**{}. {}**".format(j, [i][0]))
            j += 1
            if(j == 16):
                break
        st.image(img2, width=960)

    elif(select == 'Most Used Words in The Titles'):
        st.header("Most Common Words in Movie Titles :")
        st.markdown(
            "Words Like **Man**,** The**,**Death**,**night** etc are commonly used in movie titles.")

        st.markdown("After compiling 10000 movies' titles from 1966 to 2016,i dropped them into a wordcloud to see what were the touchstone topics in the movie industry. Here's the result:")
        img = Image.open('./assets/images/wordcloud.png')
        st.image(img)
        st.subheader("Count Pie chart")
        img2 = Image.open("./assets/images/wordcount.jpg")
        st.image(img2)
    elif(select == 'Movies With longest Runtime'):
        st.header('15 longest Movies : ')
        st.markdown(
            'Do u know there are films that goes upto 15 hours or even more.[The Story of Film: An Odyssey](https://en.wikipedia.org/wiki/The_Story_of_Film:_An_Odyssey) is one of those,a 2011 British documentary film about the history of film, presented on television in 15 one-hour chapters with a total length of over 900 minutes.')

        img = Image.open('./assets/images/odessy.jpg')
        st.image(img)
        img2 = Image.open('./assets/images/top_15_runtime.png')
        st.image(img2, width=700)

        st.subheader("Other Longest Movies are ")
        show(runtime_id, runtime_list)


if __name__ == '__main__':
    main()
