from utils.data_utils import *
from utils.pandas_utils import view_hitab_data
import ast
import pandas as pd
from datasets import Dataset, DatasetDict
from typing import Optional, List, Union

def is_empty(value : Optional[Union[List[str], str]]):
    if isinstance(value, List):
        values = [clean_text(x) for x in value]
        values = [x for x in values if len(x)>0]
        return False if len(values)>0 else True
    else:
        value = clean_text(value)
        return False if len(value)>0 else True

def clean_value(value: Optional[Union[List[str], str]]):
    if isinstance(value, List):
        list = [clean_text(x) for x in value]
        return list
    else:
        return clean_text(value)

def clean_text(text: str):    
    text = str(text)
    text = text.replace("\n", " ")
    text = text.strip().lower()
    return text

def flatten_list(nested_list):
    return [item for sublist in nested_list for item in sublist]

def update_answer_coordinates(answer_coordinates, table):

    col_headers = table["column_header"]
    measurement_unit = table["measurement_unit"]

    new_coordinates = []
    for coor in answer_coordinates:
        row_index, col_index = ast.literal_eval(coor)
        col_index_reduction = len(col_headers)
        index_reduction = col_index_reduction+1 if measurement_unit else col_index_reduction
        
        new_coordinates.append(tuple((row_index-index_reduction, col_index)))
    return new_coordinates

def answer_coordinates_in_column_headers(answer_coordinates):
    is_match = False
    for coor in answer_coordinates:
        row_index, col_index = coor
        is_match = True if row_index < 0 else False
    return is_match

def merge_colum_header(col_headers):
    if len(col_headers) == 1:
        return col_headers[0]
    elif len(col_headers) == 2:
        return [' '.join(list(dict.fromkeys(i))) for i in zip(col_headers[0], col_headers[1])]
    elif len(col_headers) == 3:
        return [' '.join(list(dict.fromkeys(i))) for i in zip(col_headers[0], col_headers[1], col_headers[2] )]
    elif len(col_headers) == 4:
        return [' '.join(list(dict.fromkeys(i))) for i in zip(col_headers[0], col_headers[1], col_headers[2] , col_headers[3] )]
    else:
        print(json.dumps(col_headers, indent=2))
        raise ValueError(f"column header > 4, size = {len(col_headers)} ")

def merge_row_header(row_headers):

    outputs = []
    for index, x in enumerate(row_headers):
        headers = []
        if x["level"]==0:
            header0 = x["value"]
            headers.append(header0)

        elif x["level"]==1:
            previous_items = row_headers[:index]
            parent_lv0 = [y for y in previous_items if y["level"]==0][-1]
            
            header0 = parent_lv0["value"]
            if parent_lv0["data_is_empty"]:
                headers.append(header0)

            header1 = x["value"]
            headers.append(header1)            
        
        elif x["level"]==2:
            previous_items1 = row_headers[:index]
            
            parent_lv1 = [(index,y) for index,y in enumerate(previous_items1) if y["level"]==1][-1]
            parent_lv1_index = parent_lv1[0]

            previous_items0 = row_headers[:parent_lv1_index]
            parent_lv0 = [(index,y) for index,y in enumerate(previous_items0) if y["level"]==0][-1]

            header0 = parent_lv0[1]["value"]
            if parent_lv0[1]["data_is_empty"]:
                headers.append(header0)

            header1 = parent_lv1[1]["value"]
            if parent_lv1[1]["data_is_empty"]:
                headers.append(header1)    

            header2 = x["value"]
            headers.append(header2)         
        
        else:
            raise ValueError(f'row header level > 3, level = {x["level"]}')
        
        headers = ' '.join(headers)
        headers = headers.split()
        headers = " ".join(sorted(set(headers), key=headers.index))
        outputs.append(headers)

    return outputs
     
class ITQA:
    def __init__(
        self, 
        questions, 
        tables, 
        flat_col_headers: bool = True,
        flat_row_headers: bool = True,
        invalid_data_ids: List[str] = None,
        lang: str = 'id'
    ):
        self.questions = questions
        self.tables = tables
        self.flat_col_headers = flat_col_headers
        self.flat_row_headers = flat_row_headers
        self.invalid_data_ids = invalid_data_ids
        self.lang = lang

        self.format_questions()
        self.format_tables()

    def get_value(self, value):
        if self.lang == 'id':
            value = value["id"] if is_empty(value["revision_id"]) else value["revision_id"]
            return clean_value(value)
        else:
            return clean_value(value["en"])

    def format_questions(self):
        print('formating questions ...')
        outputs = []
        for x in self.questions:
            table = [table for table in self.tables if table["id"] == x["table_id"] ][0]
            out = {
                "id" : x["id"],
                "question_type" : x["question_type"],
                "question" : self.get_value(x["question"]).strip(),
                "answer_coordinates" : update_answer_coordinates(x["answer_coordinates"], table),
                "answer_text" : self.get_value(x["answer"]),
                "table_id" : x["table_id"],
                "table_source" : x["table_source"],
            }
            outputs.append(out)
        self.formatted_questions = outputs

    def format_tables(self):
        print('formating tables...')
        outputs = []
        for x in self.tables:
            col_headers = []
            col_indexs = flatten_list([x["col_indexs"] for x in  x["column_header"][0]["items"]])
            col_headers_items = flatten_list([x["items"] for x in  x["column_header"]])

            
            for i in range(len(x["column_header"])):
                row=[]
                for j in col_indexs:
                    try:
                        cell = (next(item for item in col_headers_items if i in item["row_indexs"] and j in item["col_indexs"]))
                        row.append(self.get_value(cell["name"]))
                    except:
                        print(x["id"])
                col_headers.append(row)

            measurement_unit = x["measurement_unit"]
            if measurement_unit:
                row=[]
                for j in col_indexs:
                    row.append(self.get_value(measurement_unit))
                col_headers.append(row)

            if self.flat_col_headers:
                col_headers = merge_colum_header(col_headers)

            row_header_name = self.get_value(x["row_header"][0]["name"])
            if len(row_header_name)>0:
                col_headers.insert(0, row_header_name)
            else:
                col_headers.insert(0, "kharakteristik" if self.lang == 'id' else "characteristics")

            ## row header
            # print(is_empty(x["data"]['0']))
            # break
            row_headers = []

            for row in x["row_header"]:
                new_row = []
                for i, cell in enumerate(row["items"]):
                    new_row.append({
                        "level":cell["level"],
                        "value":self.get_value(cell["name"]),
                        "data_is_empty": True if is_empty(list(x["data"].values())[i]) else False
                    })
                row_headers.append(new_row)
            
            if self.flat_row_headers:
                row_headers = merge_row_header(row_headers[0])

            data = []
            for index, row in enumerate(x["data"].values()):
                row.insert(0, row_headers[index])
                data.append(row)


            out = {
                "id" : x["id"],
                "title":self.get_value(x["title"]),
                "header" : col_headers,
                "data" : data,
            }
            outputs.append(out)        
                    
        self.formatted_tables = outputs

    def generate_data(self):
        invalid_ques = []
        outputs = []
        for ques in self.formatted_questions:
            # try:
                table = [x for x in self.formatted_tables if x["id"]==ques["table_id"]][0]
                if not answer_coordinates_in_column_headers(ques["answer_coordinates"]):
                    df = pd.DataFrame(data=table["data"], columns=table["header"]).astype(str)
                    table_from_df = json.loads(df.to_json(orient="split"))
                    ques["table"] = {
                        "id":table["id"],
                        "title":table["title"],
                        "header": list(map(str, table_from_df["columns"])) if self.flat_col_headers else table["header"] , 
                        "data":table_from_df["data"]
                    }
                    del ques["table_id"]
                    outputs.append(ques)   
            
        return outputs, invalid_ques