import re
import sys
import json
import json.tool

FLAGS = re.VERBOSE | re.MULTILINE | re.DOTALL
WHITESPACE = re.compile(r'[ \t\n\r]*', FLAGS)

class ConcatJSONDecoder(json.JSONDecoder):
    def decode(self, s, _w=WHITESPACE.match):
        s_len = len(s)
        objs = []
        end = 0
        while end != s_len:
            obj, end = self.raw_decode(s, idx=_w(s, end).end())
            end = _w(s, end).end()
            objs.append(obj)
        return objs

def load_blob_data(filename):
    print('reading data file')
    sys.stdout.flush()
    with open(filename,'r') as f:
        dataList = json.load(f,cls=ConcatJSONDecoder)
    print('done')
    sys.stdout.flush()
    return dataList




