import pandas as pd
import os
import sys
import numpy as np
import nltk
import requests

processed_path = os.path.dirname(__file__) + "/../../data/processed/data.pickle"
interim_path = os.path.dirname(__file__) + "/../../data/interim/data.pickle"

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

    path = os.path.dirname(__file__) + "/../../data/external/stopord.txt"
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
        return []


if __name__ == "__main__":

    interim_df = pd.read_pickle(interim_path)
    df = interim_df.copy()

    #%% Building features
    print("Building features...")

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

    # Splitting tekst into lists
    df["Tekst"] = df["Tekst"].str.split("[! ,.\-_?()]+")

    # Stop-words
    print("Removing stop word (expect long time) ... ", end=" ")
    df["Tekst"] = df["Tekst"].apply(remove_stop_words, stopwords=get_stop_words())
    print("Removing done!")

    # Stemming
    print("Steming word (expect long time) ... ", end=" ")
    df["Tekst"] = df["Tekst"].apply(stem)
    print("Steming done!")

    #%% Pickling processed_path data
    print("Saving to pickle...", end=" ")
    df.to_pickle(processed_path)
    print("Done!")

else:

    df = pd.read_pickle(processed_path)

    #    dict(zip(*np.unique(df['Tekst'][0], return_counts=True)))
