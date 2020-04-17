#%%
import xml.etree.ElementTree as ET
import pandas as pd
import os
import sys

file_path = os.path.dirname(__file__)

#%%

# If file is directly run by python:
if __name__ == "__main__":
    df = pd.DataFrame()

    # Reading data from each file, appending each 'Tale' entry to dataframe
    for file in os.scandir(file_path + "/../../data/raw/"):

        if file.path.endswith(".xml"):
            print("Proccessing: {0}".format(file))
            et = ET.parse(file)
            df = df.append(list({x.tag: x.text for x in y} for y in et.iter("Tale")))

    # Proper datatypes, and sorting
    df["Starttid"] = pd.to_datetime(df["Starttid"])
    df["Sluttid"] = pd.to_datetime(df["Sluttid"])

    df.sort_values("Starttid", inplace=True)
    df.reset_index(inplace=True, drop=True)

    #%% Pickling
    df.to_pickle(file_path + "/../../data/interim/data.pickle")
    print("Pickle Created")

# if the file is imported:
else:
    try:
        df = pd.read_pickle(file_path + "/../../data/interim/data.pickle")
    except FileNotFoundError:
        print("Pickle not created yet - run make_dataset.py")
