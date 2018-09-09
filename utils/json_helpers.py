import codecs
import json

def load_data_from_json_file(json_file):
    data_file = codecs.open(json_file, mode='r', encoding="utf-8")
    data = json.load(data_file)
    data_file.close()

    return data

def save_data_to_json_file(data, output_json_file):
    data_file = codecs.open(output_json_file, mode='wb', encoding="utf-8")
    json.dump(data, data_file)
    data_file.close()