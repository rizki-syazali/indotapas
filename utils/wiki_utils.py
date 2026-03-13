import pandas as pd

def wiki_data_is_valid(wiki_data):
    #condition for valid data

    #first condition
    first_condition = True if wiki_data["table"]["columns"] and wiki_data["table"]["data"] else False

    #second condition : check length of row for each columns and data has same length
    second_condition =False
    unique_column_row_lengths = list(set([len(row) for row in wiki_data["table"]["columns"]]))
    unique_data_row_lengths = list(set([len(row) for row in wiki_data["table"]["columns"]]))
    if len(unique_column_row_lengths) == 1 and len(unique_data_row_lengths) == 1 and unique_column_row_lengths[0] == unique_data_row_lengths[0]:
        second_condition = True

    #third condition : columns is merged and there is no other information can be used for a column
    third_condition = True
    if wiki_data["type"] == "wikitable" and len(wiki_data["table"]["columns"])==1 and len(list(set(wiki_data["table"]["columns"][0])))==1:
        third_condition = False
    
    fourth_condition = False
    text_types = [ text["type"] for text in wiki_data["texts"]]
    if "PREV_DESCRIPTION" in text_types or "NEXT_DESCRIPTION" in text_types:
        fourth_condition = True

    #fourth_condition
    last_condition = True
    try:
        df = pd.DataFrame(data=wiki_data["table"]["data"], columns=wiki_data["table"]["columns"])
    except:
        last_condition = False    
    
    return True if first_condition and second_condition and third_condition and fourth_condition and last_condition else False

def most_frequent(List):
    return max(set(List), key = List.count, default=0)

def least_frequent(List):
    return min(set(List), key = List.count, default=0)