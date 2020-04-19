#%%
import pandas as pd
import os
import sys
import numpy as np
from func import *

processed_path = os.path.dirname(__file__) + "/../../data/processed/reduced_data.pickle"

if __name__ == "__main__":
    path = os.path.dirname(__file__)

    from build_features import df

    df = df.head(1000)

    print(
        "Removing rows where formand, aldersformanden, midlertidig formand is speaking"
    )

    df = df[
        (df["Rolle"] != "formand")
        & (df["Rolle"] != "aldersformanden")
        & (df["Rolle"] != "midlertidig formand ")
    ].reset_index(drop=True)

    # Remove instances where the text length is zero
    df = df[df["Tekst"].apply(len).values != 0].reset_index(drop=True)

    print("Constructing th global bag of words (expect long time) ... ", end=" ")
    words, count = np.unique(
        np.concatenate(df["Tekst"].to_numpy().ravel()), return_counts=True
    )
    print("Bag of words done!")

    create_additional_stopwords(words, count, word_count_threshold=25)
    print(
        "Removing not frequent words and additional manuel addeed words (expect long time) ... ",
        end=" ",
    )
    df["Tekst"] = df["Tekst"].apply(
        remove_words,
        stopwords=read_txt_file(
            path + "/../../data/external/additional_word_to_be_removed.txt"
        ),
    )
    print("Removing done!")

    # Remove instances where the text length is zero
    # happens again after not frequent word is removed.
    df = df[df["Tekst"].apply(len).values != 0].reset_index(drop=True)

    #%% Pickling processed_path data
    print("Saving to pickle...", end=" ")
    df.to_pickle(processed_path)
    print("Done!")


# %%
