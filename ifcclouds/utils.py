import json

def read_ply(file):
  with open(file, 'r') as f:
    lines = f.readlines()
    # remove header
    lines = lines[8:]
    return lines
  

def load_classes_from_json(json_file_path, verbose=False):
    if verbose: print('Loading classes from %s' % json_file_path)
    try:
        with open(json_file_path) as json_file:
            return json.load(json_file)
    except Exception as e:
        if verbose: print(e)
        print('Error loading classes from %s, return default' % json_file_path)
        return None