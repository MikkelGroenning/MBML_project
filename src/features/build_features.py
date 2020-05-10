import pandas as pd
import os
import sys 
import numpy as np
import nltk
import requests
import time
import scipy
from pathlib import Path

import pickle
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

sys.path.append( str(Path(__file__).parent.resolve() / '..') )
from data.make_dataset import df as interim_df

processed_path = Path(__file__).parent.resolve()/ ".." / ".." / "data" / "processed"

# From https://stackoverflow.com/questions/36182502/add-stemming-support-to-countvectorizer-sklearn
danish_stemmer = nltk.stem.SnowballStemmer('danish')
class StemmedCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        analyzer = super(StemmedCountVectorizer, self).build_analyzer()
        return lambda doc: ([danish_stemmer.stem(w) for w in analyzer(doc)])

# Consider JSON
name_party_dict = {
    'Niels Helveg Petersen' : 'RV',
    'Lars Løkke Rasmussen' : 'V',
    'Eva Kjer Hansen' : 'V',
    'Claus Hjort Frederiksen' : 'V',
    'Troels Lund Poulsen' : 'V',
    'Karen Ellemann' : 'V',
    'Holger K. Nielsen' : 'SF',
    'Inger Støjberg' : 'V',
    'Birthe Rønn Hornbech' : 'V',
    'Helge Sander' : 'V',
    'Mogens Lykketoft' : 'S',
    'Helge Adam Møller' : 'KF',
    'Søren Espersen' : 'DF',
    'Brian Mikkelsen' : 'KF',
    'Lars Barfoed' : 'KF',
    'Kristian Jensen' : 'V',
    'Søren Gade' : 'V',
    'Carina Christensen' : 'KF',
    'Jens Vibjerg' : 'V',
    'Karen J. Klint' : 'S',
    'Bertel Haarder' : 'V',
    'Lene Espersen' : 'KF',
    'Per Stig Møller' : 'KF',
    'Connie Hedegaard' : 'KF',
    'Jakob Axel Nielsen' : 'KF',
    'Bent Bøgsted' : 'DF',
    'Pernille Frahm' : 'SF',
    'Lykke Friis' : 'V',
    'Rasmus Jarlov (KF)' : 'KF',
    'Ulla Tørnæs' : 'V',
    'Irene Simonsen (V)' : 'V',
    'Gitte Lillelund Bech' : 'V',
    'Benedikte Kiær' : 'KF',
    'Tina Nedergaard' : 'V',
    'Charlotte Sahl-Madsen' : 'KF',
    'Henrik Høegh' : 'V',
    'Hans Christian Schmidt' : 'V',
    'Søren Pind' : 'V',
    'Peter Christensen' : 'V',
    'Marianne Jelved' : 'RV',
    'Lars Christian Lilleholt' : 'V',
    'Thor Möger Pedersen' : 'SF',
    'Karen Hækkerup' : 'S',
    'Bjarne Corydon' : 'S',
    'Henrik Dam Kristensen' : 'S',
    'Mette Frederiksen' : 'S',
    'Astrid Krag' : 'SF',
    'Manu Sareen' : 'RV',
    'Christian Friis Bach' : 'RV',
    'Uffe Elbæk' : 'RV',
    'Ole Sohn' : 'S',
    'Morten Bødskov' : 'S',
    'Helle Thorning-Schmidt' : 'S',
    'Margrethe Vestager' : 'RV',
    'Villy Søvndal' : 'SF',
    'Morten Østergaard' : 'RV',
    'Carsten Hansen' : 'S',
    'Christine Antorini' : 'S',
    'Pia Olsen Dyhr' : 'SF',
    'Ida Auken' : 'RV',
    'Mette Gjerskov' : 'S',
    'Nicolai Wammen' : 'S',
    'Nick Hækkerup' : 'S',
    'John Dyrby Paulsen' : 'S',
    'Martin Lidegaard' : 'RV',
    'Pia Kjærsgaard' : 'DF',
    'Annette Vilhelmsen' : 'SF',
    'Anne Baastrup' : 'SF',
    'Camilla Hersom' : 'RV',
    'Henrik Sass Larsen' : 'S',
    'Jonas Dahl' : 'SF',
    'Dan Jørgensen' : 'S',
    'Sofie Carsten Nielsen' : 'RV',
    'Steen Gade' : 'SF',
    'Magnus Heunicke' : 'S',
    'Rasmus Helveg Petersen' : 'RV',
    'Kirsten Brosbøl' : 'S',
    'Mogens Jensen' : 'S',
    'Benny Engelbrecht' : 'S',
    'Lone Loklindt' : 'RV',
    'Per Clausen' : 'EL',
    'Ane Halsboe-Jørgensen' : 'S',
    'Mette Bock' : 'LA',
    'Erling Bonnesen' : 'V',
    'Karsten Lauritzen' : 'V',
    'Jørn Neergaard Larsen' : 'V',
    'Stine Brix' : 'EL',
    'Kristian Pihl Lorentzen' : 'V',
    'Ellen Trane Nørby' : 'V',
    'Sophie Løhde' : 'V',
    'Esben Lunde Larsen' : 'V',
    'Christian Juhl' : 'EL',
    'Alex Ahrendtsen' : 'DF',
    'Mai Mercado' : 'KF',
    'Simon Emil Ammitzbøll' : 'LA',
    'Søren Pape Poulsen' : 'KF',
    'Merete Riisager' : 'LA',
    'Ole Birk Olesen' : 'LA',
    'Carsten Kudsk (DF)' : 'DF',
    'Leif Mikkelsen' : 'LA',
    'Thyra Frank' : 'LA',
    'Anders Samuelsen' : 'LA',
    'DF' : 'DF',
    'KF' : 'KF'
}

def stem(word_array):
    stemmer = nltk.stem.SnowballStemmer('danish')
    
    try:
        return( [stemmer.stem(word) for word in word_array ] )
    except TypeError:
        return([])

def read_txt_file(path_to_file):

    f = open(path_to_file, 'r')
    words = f.read().splitlines()
    f.close()

    return words

def get_stop_words():

    path = Path(__file__).parent.resolve() / ".." / ".." / "data" / "external" / "stopord.txt"
    try:
        stopwords = read_txt_file(path)
    except FileNotFoundError:

        print("Downloading stopwords from Github")
        url = 'https://gist.githubusercontent.com/berteltorp/0cf8a0c7afea7f25ed754f24cfc2467b/raw/305d8e3930cc419e909d49d4b489c9773f75b2d6/stopord.txt'
        r = requests.get(url)
        with open(path, 'wb') as f:
            f.write(r.content)
        stopwords = read_txt_file(path)

    # Adding stemmed stopwords (fix from error message, sue me)
    stopwords += ['bl', 'ca', 'eks', 'pga']

    return(stopwords)

def remove_stop_words(words, stopwords):  

    try:           
        return [ x for x in words if x not in stopwords ]
    except TypeError:
        return words

def delete_cols_csr(mat, indices):
    """
    Remove the rows denoted by ``indices`` form the CSR sparse matrix ``mat``.
    """
    if not isinstance(mat, scipy.sparse.csr_matrix):
        raise ValueError("works only for CSR format -- use .tocsr() first")
    indices = list(indices)
    mask = np.ones(mat.shape[1], dtype=bool)
    mask[indices] = False
    return mat[:,mask]


if __name__ == '__main__':

    t = time.time()

    df = interim_df

    # Fjern speaker
    print('Removing speakers...')
    tmp = sum(df.Rolle.str.contains("formand"))
    df = df.drop( df.loc[ df['Rolle'].str.contains("formand") ].index )
    df.reset_index(inplace=True, drop=True)

    print(f'Removed {tmp} observations...')

    #%% Adding additional features
    print('Adding additional data...')

    # Filling party in missing observation
    df.Parti = df.Parti.fillna(df.Navn.map(name_party_dict))

    df['Navn'] = df['Navn'].str.replace(r" \(.*\)","")

    # Tilføj regering
    conditions = [\
        (df['Starttid'] <= pd.Timestamp(2011, 10, 3)),\
        ((df['Starttid'] > pd.Timestamp(2011, 10, 3)) & (df['Starttid'] <= pd.Timestamp(2014, 2, 3))),\
        ((df['Starttid'] > pd.Timestamp(2014, 2, 3)) & (df['Starttid'] <= pd.Timestamp(2015, 6, 28))),\
        ((df['Starttid'] > pd.Timestamp(2015, 6, 28)) & (df['Starttid'] <= pd.Timestamp(2016, 11, 28))),\
        (df['Starttid'] > pd.Timestamp(2016, 11, 28))\
    ]
    choices = ['0Lars1', 'Helle1', 'Helle2', 'Lars2', 'Lars3']
    df['Regering'] = np.select(conditions, choices, default=None)


    # Er parti i regering
    govs = [['V','KF'],['S','RV','SF'],['S','RV'],['V'],['V','KF','LA']]
    government_dict = dict(zip(choices, govs))
    df['I_Regering'] = [i in j for i, j in zip(df.Parti, df['Regering'].map(government_dict))]

    # Merger Nordatlantiske partier
    NApartier = ['IA','SIU', 'JF', 'T', 'SP', 'TF']
    df = df.replace(to_replace=NApartier,value='NA')

    # Vectorizing speeches   
    print('Vectorizing speeches...')
    vectorizer = StemmedCountVectorizer(
        min_df=10, 
        analyzer="word", 
        stop_words=get_stop_words()
    )
    X = vectorizer.fit_transform(df['Tekst'])
    vocabulary = vectorizer.vocabulary_

    # Remove common words
    print("Removing common words")
    tmp = pd.DataFrame(data = X, index = df.Dagsordenpunkt, columns = ['kage'])
    cancer = np.empty((df.Dagsordenpunkt.nunique(),X.shape[1]))
    for i, punkt in enumerate(df.Dagsordenpunkt.unique()):
        if (i % 732) == 0:
            print(i/df.Dagsordenpunkt.nunique()*100, "%", end=' ')
        
        try:
            cancer[i,:] = np.minimum(1,np.asarray(sum([s for s in tmp.loc[punkt].kage.values]).todense()).reshape(-1,))
        except AttributeError:
            cancer[i,:] = np.minimum(1,np.asarray(tmp.loc[punkt].kage.todense()).reshape(-1))

    
    indexes = np.where(cancer.mean(axis=0) > 0.4)[0]
    X = delete_cols_csr(X, indexes)

    print("\nFixing vocabulary")
    tmp = dict(sorted({v: k for k, v in vocabulary.items()}.items()))
    for idx in indexes:
        del tmp[idx]

    vocabulary = {word: idx for idx, word in enumerate(list(tmp.values()))}


    print("Doing TD-IDF")
    tfidf = TfidfTransformer()
    X_tfidf = tfidf.fit_transform(X)

    
    print("Creating corpus")
    cx = scipy.sparse.coo_matrix(X)
    corpus = [[] for _ in range(X.shape[0])]
    for speech, col, data in zip(cx.row, cx.col, cx.data):
        corpus[speech].append((col, data))

    cx_tfidf = scipy.sparse.coo_matrix(X_tfidf)
    corpus_tfidf = [[] for _ in range(X_tfidf.shape[0])]
    for speech, col, data in zip(cx_tfidf.row, cx_tfidf.col, cx_tfidf.data):
        corpus_tfidf[speech].append((col, data))


    print("Saving dataframe to pickle...")
    df.to_pickle( processed_path / "data.pickle")

    print("Saving vectorized data to pickle...")
    with open( processed_path / "vectorized.pickle", 'wb' ) as f:
        pickle.dump( (vocabulary, X, X_tfidf, corpus, corpus_tfidf), f )

    print("Done! (Took %.2f seconds)" % (time.time() - t))

else:

    df = pd.read_pickle( processed_path / "data.pickle")

    with open(processed_path / "vectorized.pickle", 'rb') as f:
        vocabulary, X, X_tfidf, corpus, corpus_tfidf = pickle.load(f)