import pandas as pd
import os
import sys
import numpy as np
import nltk
import requests
import time
from pathlib import Path
import tqdm as tqdm
import pickle
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

sys.path.append(str(Path(__file__).parent.resolve() / ".."))

from data.make_dataset import df as interim_df

processed_path = Path.cwd() / Path(r"data/processed")

# From https://stackoverflow.com/questions/36182502/add-stemming-support-to-countvectorizer-sklearn
danish_stemmer = nltk.stem.SnowballStemmer("danish")


class StemmedCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        analyzer = super(StemmedCountVectorizer, self).build_analyzer()
        return lambda doc: ([danish_stemmer.stem(w) for w in analyzer(doc)])


# Consider JSON
name_party_dict = {
    "Niels Helveg Petersen": "RV",
    "Lars Løkke Rasmussen": "V",
    "Eva Kjer Hansen": "V",
    "Claus Hjort Frederiksen": "V",
    "Troels Lund Poulsen": "V",
    "Karen Ellemann": "V",
    "Holger K. Nielsen": "SF",
    "Inger Støjberg": "V",
    "Birthe Rønn Hornbech": "V",
    "Helge Sander": "V",
    "Mogens Lykketoft": "S",
    "Helge Adam Møller": "KF",
    "Søren Espersen": "DF",
    "Brian Mikkelsen": "KF",
    "Lars Barfoed": "KF",
    "Kristian Jensen": "V",
    "Søren Gade": "V",
    "Carina Christensen": "KF",
    "Jens Vibjerg": "V",
    "Karen J. Klint": "S",
    "Bertel Haarder": "V",
    "Lene Espersen": "KF",
    "Per Stig Møller": "KF",
    "Connie Hedegaard": "KF",
    "Jakob Axel Nielsen": "KF",
    "Bent Bøgsted": "DF",
    "Pernille Frahm": "SF",
    "Lykke Friis": "V",
    "Rasmus Jarlov (KF)": "KF",
    "Ulla Tørnæs": "V",
    "Irene Simonsen (V)": "V",
    "Gitte Lillelund Bech": "V",
    "Benedikte Kiær": "KF",
    "Tina Nedergaard": "V",
    "Charlotte Sahl-Madsen": "KF",
    "Henrik Høegh": "V",
    "Hans Christian Schmidt": "V",
    "Søren Pind": "V",
    "Peter Christensen": "V",
    "Marianne Jelved": "RV",
    "Lars Christian Lilleholt": "V",
    "Thor Möger Pedersen": "SF",
    "Karen Hækkerup": "S",
    "Bjarne Corydon": "S",
    "Henrik Dam Kristensen": "S",
    "Mette Frederiksen": "S",
    "Astrid Krag": "SF",
    "Manu Sareen": "RV",
    "Christian Friis Bach": "RV",
    "Uffe Elbæk": "RV",
    "Ole Sohn": "S",
    "Morten Bødskov": "S",
    "Helle Thorning-Schmidt": "S",
    "Margrethe Vestager": "RV",
    "Villy Søvndal": "SF",
    "Morten Østergaard": "RV",
    "Carsten Hansen": "S",
    "Christine Antorini": "S",
    "Pia Olsen Dyhr": "SF",
    "Ida Auken": "RV",
    "Mette Gjerskov": "S",
    "Nicolai Wammen": "S",
    "Nick Hækkerup": "S",
    "John Dyrby Paulsen": "S",
    "Martin Lidegaard": "RV",
    "Pia Kjærsgaard": "DF",
    "Annette Vilhelmsen": "SF",
    "Anne Baastrup": "SF",
    "Camilla Hersom": "RV",
    "Henrik Sass Larsen": "S",
    "Jonas Dahl": "SF",
    "Dan Jørgensen": "S",
    "Sofie Carsten Nielsen": "RV",
    "Steen Gade": "SF",
    "Magnus Heunicke": "S",
    "Rasmus Helveg Petersen": "RV",
    "Kirsten Brosbøl": "S",
    "Mogens Jensen": "S",
    "Benny Engelbrecht": "S",
    "Lone Loklindt": "RV",
    "Per Clausen": "EL",
    "Ane Halsboe-Jørgensen": "S",
    "Mette Bock": "LA",
    "Erling Bonnesen": "V",
    "Karsten Lauritzen": "V",
    "Jørn Neergaard Larsen": "V",
    "Stine Brix": "EL",
    "Kristian Pihl Lorentzen": "V",
    "Ellen Trane Nørby": "V",
    "Sophie Løhde": "V",
    "Esben Lunde Larsen": "V",
    "Christian Juhl": "EL",
    "Alex Ahrendtsen": "DF",
    "Mai Mercado": "KF",
    "Simon Emil Ammitzbøll": "LA",
    "Søren Pape Poulsen": "KF",
    "Merete Riisager": "LA",
    "Ole Birk Olesen": "LA",
    "Carsten Kudsk (DF)": "DF",
    "Leif Mikkelsen": "LA",
    "Thyra Frank": "LA",
    "Anders Samuelsen": "LA",
    "DF": "DF",
    "KF": "KF",
}


def stem(word_array):
    stemmer = nltk.stem.SnowballStemmer("danish")

    try:
        return [stemmer.stem(word) for word in word_array]
    except TypeError:
        return []


def read_txt_file(path_to_file):

    f = open(path_to_file, "r")
    words = f.read().splitlines()
    f.close()

    return words


def get_stop_words():

    path = Path.cwd() / Path(r"data/external/additional_word_to_be_removed.txt")
    try:
        stopwords = read_txt_file(path)
    except FileNotFoundError:

        print("Downloading stopwords from Github")
        url = "https://gist.githubusercontent.com/berteltorp/0cf8a0c7afea7f25ed754f24cfc2467b/raw/305d8e3930cc419e909d49d4b489c9773f75b2d6/stopord.txt"
        r = requests.get(url)
        with open(path, "wb") as f:
            f.write(r.content)
        stopwords = read_txt_file(path)
    return stopwords


def remove_stop_words(words, stopwords):

    try:
        return [x for x in words if x not in stopwords]
    except TypeError:
        return words


if __name__ == "__main__":

    t = time.time()

    df = interim_df

    # Fjern speaker
    print("Removing speakers...")
    tmp = sum(df.Rolle.str.contains("formand"))
    df = df.drop(df.loc[df["Rolle"].str.contains("formand")].index)
    df.reset_index(inplace=True, drop=True)

    print(f"Removed {tmp} observations...")

    #%% Adding additional features
    print("Adding additional data...")

    # Filling party in missing observation
    df.Parti = df.Parti.fillna(df.Navn.map(name_party_dict))

    df["Navn"] = df["Navn"].str.replace(r" \(.*\)", "")

    # Tilføj regering
    conditions = [
        (df["Starttid"] <= pd.Timestamp(2011, 10, 3)),
        (
            (df["Starttid"] > pd.Timestamp(2011, 10, 3))
            & (df["Starttid"] <= pd.Timestamp(2014, 2, 3))
        ),
        (
            (df["Starttid"] > pd.Timestamp(2014, 2, 3))
            & (df["Starttid"] <= pd.Timestamp(2015, 6, 28))
        ),
        (
            (df["Starttid"] > pd.Timestamp(2015, 6, 28))
            & (df["Starttid"] <= pd.Timestamp(2016, 11, 28))
        ),
        (df["Starttid"] > pd.Timestamp(2016, 11, 28)),
    ]
    choices = ["0Lars1", "Helle1", "Helle2", "Lars2", "Lars3"]
    df["Regering"] = np.select(conditions, choices, default=None)

    # Er parti i regering
    govs = [["V", "KF"], ["S", "RV", "SF"], ["S", "RV"], ["V"], ["V", "KF", "LA"]]
    government_dict = dict(zip(choices, govs))
    df["I_Regering"] = [
        i in j for i, j in zip(df.Parti, df["Regering"].map(government_dict))
    ]

    # Merger Nordatlantiske partier
    NApartier = ["IA", "SIU", "JF", "T", "SP", "TF"]
    df = df.replace(to_replace=NApartier, value="NA")

    # Vectorizing speeches
    print("Vectorizing speeches...")
    vectorizer = StemmedCountVectorizer(
        min_df=10, analyzer="word", stop_words=get_stop_words()
    )
    X = vectorizer.fit_transform(df["Tekst"])
    vocabulary = vectorizer.vocabulary_
    tfidf = TfidfTransformer()
    X_tfidf = tfidf.fit_transform(X)

    print("Saving dataframe to pickle...")
    df.to_pickle(processed_path / "data.pickle")

    print("Saving vectorized data to pickle...")
    with open(str(processed_path / "vectorized.pickle"), "wb") as f:
        pickle.dump((vocabulary, X, X_tfidf), f)

    print("Done! (Took %.2f seconds)" % (time.time() - t))

else:
    df = pd.read_pickle(processed_path / "data.pickle")

    with open(processed_path / "vectorized.pickle", "rb") as f:
        vocabulary, X, X_tfidf = pickle.load(f)
