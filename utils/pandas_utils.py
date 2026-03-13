import pandas as pd
from IPython.display import display
def highlight_answer_coordinates(s,columns, answer_coordinates):
    answer_coordinates = list(tuple(sub) for sub in answer_coordinates)
    col_index = columns.index(s.name)
    s_coord = pd.Series([(i, col_index) for i, v in s.items()])
    match = s_coord.isin(answer_coordinates)
    return ['background-color: yellow;color:black' if v else '' for v in match]

def highlight_answer_texts(s, answer_text):
    match = s.isin(answer_text)
    return ['background-color: #0c68fe;color:white' if v else '' for v in match]

def view_hitab_data(item):

    table = item["table"]
    # columns=table["header"]
    # columns= list(map(tuple, table["header"]))
    # columns= pd.MultiIndex.from_tuples(columns)
    df = pd.DataFrame(data = table["data"], columns=table["header"]).astype(str)
    # df = pd.DataFrame(table["data"], columns=columns)
    df = df.style.apply(lambda x : highlight_answer_coordinates(x,df.columns.tolist(), item["answer_coordinates"])) \
                #  .apply(lambda x : highlight_answer_texts(x, item["answer_text"]))

    
    print(f'id \t\t\t= {item["id"]}')
    print(f'question \t\t= {item["question"]}')
    print(f'answer_coordinates \t= {item["answer_coordinates"]}')
    print(f'answer_text \t\t= {item["answer_text"]}')
    print(f'question_type \t\t= {item["question_type"]}')
    print(f'table_id \t\t= {table["id"]}')
    print(f'table_source \t\t= {item["table_source"]}')
    print(f'title \t\t\t= {table["title"]}')
    display(df)
