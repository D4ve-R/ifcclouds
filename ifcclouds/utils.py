import json
from plyfile import PlyData
  

def read_ply(file):
   plydata = PlyData.read(file)
   return plydata['vertex'].data
  

def load_classes_from_json(json_file_path, verbose=False):
    if verbose: print('Loading classes from %s' % json_file_path)
    try:
        with open(json_file_path) as json_file:
            return json.load(json_file)
    except Exception as e:
        if verbose: print(e)
        print('Error loading classes from %s, return default' % json_file_path)
        return None