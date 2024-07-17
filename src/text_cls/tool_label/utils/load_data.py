from typing import Text, Any
from json import loads, load, dump

def read_json_line(path: Text):
    with open(path, 'r', encoding= 'utf-8') as f:
        lst = list(f)

    res = {
        '__count__': len(lst),
        'data': []
    }
    for e in lst:
        tmp_dict = loads(e)
        res['data'].append(tmp_dict)
    return res

def read_json_file(path: Text):
    with open(path, 'r', encoding= 'utf-8') as f:
        res = load(f)
    return res

def write_json_file(path: Text, json_obj: Any)-> None:
    with open(path, 'w', encoding= 'utf-8') as f:
        dump(
        obj = json_obj,
            fp=f,
            indent= 4
        )
    
if __name__ == "__main__":
    data = read_json_line(
        path = 'data\elementary.jsonl'
    )

    write_json_file(
        path= 'data\elementary.json',
        json_obj= data
    )
