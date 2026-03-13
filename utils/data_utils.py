import unicodedata
import re
import pickle
import json
from bs4 import Tag

def read_pickle(filepath):
    with open(filepath, "rb") as f:
        return pickle.load(f)

def save_to_pickle(output_file, data):
    with open(output_file, "wb") as f:
        pickle.dump(data, f)

def read_json(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def save_to_json(output_file, data):
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=3)

def normalize_unicode(text):
    string = unicodedata.normalize('NFKD', text)
    # string = string.encode('ascii', errors='ignore')
    # string = str(string, encoding='ascii')
    string = string.encode('utf-8')
    string = str(string, encoding='utf-8')
    return string

def remove_citation(text):
    return re.sub(r'\[\d*\]','',text)

def remove_extra_space(text):
    text = re.sub(r'(?=.*)\s\,\s(?=.*)',', ',text)
    text = re.sub(r'\s{2,}',' ',text)
    return text.strip()

def remove_extra_comma(text):
    text = re.sub(r'(?<=[^\d\w]),', ' ',text)
    return text.strip()

def merge_newline(text: str):
    # val = text.split('\n')
    # val = ', '.join(val)
    text = re.sub(r'(?<=[\d\w])\n', ', ',text)
    text = re.sub(r'(?<=[^\d\w])\n', ' ',text)
    return text

def encode_dash_unicode_date(text):
    dash_code = '\u2013'
    months = ['januari', 'februari', 'maret', 'april','mei','juni','juli','agustus', 'september', 'oktober', 'november', 'desember']
    months_in_string = [x for x in months if x in text.lower()]
    text_has_month = True if len(months_in_string)>0 else False

    do_replace = False
    if text_has_month:
        do_replace = True
        
    else:
        regexp1 = re.compile(r'\d{1,4}\s*[\u2013]\s*sekarang')
        regexp2 = re.compile(r'\d{1,4}\s*[\u2013]\s*\d{1,4}')
        if regexp1.search(text) or regexp2.search(text):
            do_replace = True

    if do_replace:
        return re.sub(dash_code,' - ',text)
    else:
        return text

def clean_text(text):
    text = encode_dash_unicode_date(text)
    text = remove_citation(text)
    text = normalize_unicode(text)
    # text = merge_newline(text)
    text = remove_extra_comma(text)
    text = remove_extra_space(text)
    text = text.replace('\u2013',' - ')
    return text

def clean_cell(cell: Tag):
        for span in cell.find_all('span'):
            span.unwrap()
        
        for x in cell.find_all(['a','abbr']):
            x.unwrap()
        
        for ul in cell.find_all('ul'):
            for li in  ul.find_all('li'):
                li.insert(1, '\n')
                li.unwrap()
            ul.unwrap()
        
        for br in cell.find_all('br'):
            br.replace_with('\n')

        text = cell.text.strip().split('\n')
        text = [x.strip() for x in text if x.strip()]
        text = ', '.join(text)
        text = clean_text(text)
        return text


def is_empty_list(list):
    return True if all(s is None or s=='' or len(str(s))==0 for s in list) else False

def has_same_length(array):
    return len(list(set([len(x) for x in array])))==1
