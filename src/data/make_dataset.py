#%%
import xml.etree.ElementTree as ET
import pandas as pd
import os
import sys

file_path = os.path.dirname(__file__)


def tale_iterator(et):
    current_meeting_id = None
    for element in et.iter():
        if element.tag == "Tale":
            observation = {x.tag: x.text for x in element}
            observation.update({"MeetingId": current_meeting_id})
            yield observation
        elif element.tag == "Møde":
            try:
                current_meeting_id = next(
                    (x for x in element if x.tag == "MeetingId")
                ).text
            except StopIteration:
                raise Warning("No meeting category found")


# If file is directly run by python:
if __name__ == "__main__":
    df = pd.DataFrame()

    # Reading data from each file, appending each 'Tale' entry to dataframe
    for file in os.scandir(file_path + "/../../data/raw/"):

        if file.path.endswith(".xml"):
            print("Proccessing: {0}".format(file.name))
            et = ET.parse(file)
            df = df.append(list(tale_iterator(et)))

    # Proper datatypes, and sorting
    df["Starttid"] = pd.to_datetime(df["Starttid"])
    df["Sluttid"] = pd.to_datetime(df["Sluttid"])

    df.sort_values("Starttid", inplace=True)
    df.reset_index(inplace=True, drop=True)

    #%% Pickling
    print("Saving to pickle...", end=" ")
    df.to_pickle(file_path + "/../../data/interim/data.pickle")
    print("Pickle Created")

# if the file is imported:
else:
    try:
        df = pd.read_pickle(file_path + "/../../data/interim/data.pickle")
    except FileNotFoundError:
        print("Pickle not created yet - run make_dataset.py")


# %%
