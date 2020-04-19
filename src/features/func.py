from nltk.stem import SnowballStemmer

import os
import requests
import numpy as np


def stem(word_array):
    stemmer = SnowballStemmer("danish")

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


def remove_words(words, stopwords):
    try:
        return [x for x in words if x not in stopwords]
    except TypeError:
        return []


def create_additional_stopwords(words, word_count, word_count_threshold=50):
    path = (
        os.path.dirname(__file__)
        + "/../../data/external/additional_word_to_be_removed.txt"
    )

    manuel_word_list = [
        "jeg",
        "",
        "-",
        "altså",
        "så",
        "der",
        "i",
        "a",
        "f",
        "000",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
    ]

    count_sorted = np.argsort(-word_count)
    words = words[count_sorted]
    word_count = word_count[count_sorted]

    words_below_threshold = words[count_sorted <= word_count_threshold].tolist()

    words_not_to_include = manuel_word_list + words_below_threshold

    with open(path, "w") as f:
        for item in words_not_to_include:
            f.write("%s\n" % item)
