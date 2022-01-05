import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string

print("We are now computing")
anime = pd.read_csv('Anime_data.csv', encoding='latin')
#print('anime (shape):', anime.shape)#anime (shape): (17002, 15) shape will return a tuple of the dimentionality 
anime.head()#Pandas head() method is used to return top n (5 by default) rows of a data frame or series.

#print(anime[['Title', 'Rating', 'Producer', 'Studio']].loc[anime['Type'] == 'Movie'])

#we willuse text_cleaning clean the text
#re.sub() will replace a substring with a different substring
def text_cleaning(text):
    text = re.sub(r'&quot;', '', text)
    text  = "".join([char for char in text if char not in string.punctuation])
    text = re.sub(r'.hack//', '', text)
    text = re.sub(r'&#039;', '', text)
    text = re.sub(r'A&#039;s', '', text)
    text = re.sub(r'I&#039;', 'I\'', text)
    text = re.sub(r'&amp;', 'and', text)
    text = re.sub(r'Â°', '',text)
    
    return text
anime['Title'] = anime['Title'].apply(text_cleaning)# this will get clean the strings for all the Title
#some of the data in csv is not filled
#isnull() will return True to empty cell, and True to filled cell
#isnull().sum() will return a list of null counts for each column header
anime.isnull().sum()
anime.describe()# this will analyze the data
C = anime['Rating'].mean()#store the mean of the rating
m = anime['ScoredBy'].quantile(0.85)#store the quantile of ScoreBy
#.copy data then  loc[] return row as series,loc[[]] returns dataframe
q_animes = anime.copy().loc[anime['ScoredBy'] >= m]#exclude anime with low ScoreBy
#q_animes = anime.copy().loc[anime['ScoredBy']]

def weighted_rating(x, m=m, C=C):
    v = x['ScoredBy']
    R = x['Rating']
    # Calculation based on the IMDB formula
    return (v/(v+m) * R) + (m/(m+v) * C)

#Create new column 'Score'.This will calculate score
# by applying weighted_rating(), it uses Scoreby & Rating
q_animes['Score'] = q_animes.apply(weighted_rating, axis=1) 
#sort data, high Score go up
q_animes = q_animes.sort_values('Score', ascending=False)
#show top 15, but only include Title', 'ScoredBy', 'Rating', 'Score
q_animes[['Title', 'ScoredBy', 'Rating', 'Score']].head(15)#way for top 15 recommend anime

#
# plt.figure(figsize=(12, 3), dpi=100)
best_score = q_animes.sort_values(by=['Score'], ascending=False)[:10]
# g = sns.barplot(best_score["Title"], best_score['Score'], palette="spring_r")
# plt.ylabel("Score", color = 'b')
# plt.xticks(rotation=45, horizontalalignment='right', color = 'b')
# plt.title('Really good animes', fontweight='bold', fontsize=15, color = 'b')
best_scores = best_score[['Score','Title','Genre', 'Studio', 'Type']].set_index('Title')
print("Method 1")
print(best_scores)#way for top 10 recommend anime

#(1) Content Based filtering
#get_recommendations()->return 15 anime
anime['Synopsis'].isnull().sum()#number of anime with no synopsis
anime['Synopsis'] = anime['Synopsis'].fillna('')#fill all the Nall with spaces

#now we import several more libraries, and crate some variable
#pip install sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
#Term(word) Frequency — Inverse Document(sentence) Frequency
#fit_transform: Learn vocabulary and idf, return document-term matrix.
tfidf = TfidfVectorizer(stop_words='english')
tfidf
tfidf_matrix = tfidf.fit_transform(anime['Synopsis'])
tfidf_matrix
#tfidf_matrix.shape

#???
#Linear Kernel is used when data can be separated using a single Line
#we mostly use Linear Kernel in Text Classification.
from sklearn.metrics.pairwise import linear_kernel
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
cosine_sim

#Get all anime titles
indices = pd.Series(anime.index, index=anime['Title']).drop_duplicates()
indices #index of all title
indices['Cowboy Bebop']#return 0 because it is on the top
indices['Cowboy Bebop Tengoku no Tobira']#return 1


def get_recommendations(title, cosine_sim=cosine_sim):
    
    idx = indices[title]#get the index of target anime

    #run cosine_sim on target anime
    #this will return a list of similarity scores
    sim_scores = list(enumerate(cosine_sim[idx]))

    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)#sort it

    # Get the scores of the 15 most similar movies
    sim_scores = sim_scores[1:16]
    
    movie_indices = [i[0] for i in sim_scores]

    return anime['Title'].iloc[movie_indices]

print("Method 2")
print(get_recommendations('Sen to Chihiro no Kamikakushi'))#this will return top 15 similar anime based on the Synopsis.

#3)recommendation system based on producer, related genres and the studio.
features = ['Genre','Producer', 'Studio']
anime[features]#list of data with column ['Genre','Producer', 'Studio']
anime[features] = anime[features].fillna('[' ']')#fill in the null cells
#aplly literal_eval to all features
#Abstract Syntax Tree. 
#this will The literal_eval safely evaluate an 
# expression node or a string containing a Python literal or container display. 
from ast import literal_eval
for feature in features:
    anime[feature] = anime[feature].apply(literal_eval)


def clean_data(x):
    if isinstance(x, list):
        return [str.lower(i.replace(" ","")) for i in x]
    
    else:
        if isinstance(x, str):
            return str.lower(x.replace(" ",""))
        else:
            return ""    

features = ['Genre','Producer', 'Studio', 'Type']#add type

#apply clean_data to all column in features
for feature in features:
    anime[feature] = anime[feature].apply(clean_data)  



def create_soup(x):
    return " ".join(x['Genre']) + " " + x['Type'] + " " + " ".join(x['Producer']) + " " + " ".join(x['Studio']) + " " + x['Synopsis'] + " " + " ".join(x['Studio'])              


#create new column soup
anime['soup'] = anime.apply(create_soup, axis=1)    

anime['soup']

#we use the CountVectorizer() instead of TF-IDF.
#  This is because we do not want to down-weight the
#  presence of an producer if he or she has acted or
#  directed in relatively more movies. It doesn't make much intuitive sense.
from sklearn.feature_extraction.text import CountVectorizer
#CountVectorize Convert a collection of text documents to a matrix of token counts.
#fit_transform: Learn vocabulary and idf, return document-term matrix.
count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(anime['soup'])

from sklearn.metrics.pairwise import cosine_similarity

cosine_sim2 = cosine_similarity(count_matrix, count_matrix)
anime = anime.reset_index()
indices = pd.Series(anime.index, index=anime['Title'])
print("Method 3")
print(get_recommendations('Cowboy Bebop', cosine_sim2))