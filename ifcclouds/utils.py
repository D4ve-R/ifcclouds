import json
import numpy as np
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

def array_to_ply(data: np.ndArray):
    """Convert a numpy array to PLY format.
    data is a numpy array of shape (n, 4) where n is the number of vertices.
    where the fourth column is the class.
    Returns a string.
    """
    ply = 'ply\n'
    ply += 'format ascii 1.0\n'
    ply += 'element vertex %d\n' % data.shape[0]
    ply += 'property float x\n'
    ply += 'property float y\n'
    ply += 'property float z\n'
    ply += 'property int class\n'
    ply += 'end_header\n'
    for vertex in data:
        ply += '%f %f %f %d\n' % (vertex[0], vertex[1], vertex[2], vertex[3])
    return ply

def array_to_xyzrbg(data: np.ndArray):
    """Convert a numpy array to XYZRGB format.
    data is a numpy array of shape (n, 7) where n is the number of vertices.
    where the fourth column is the class.
    Returns a string.
    """
    xyzrgb = ''
    for vertex in data:
        xyzrgb += '%f %f %f %d %d %d\n' % (vertex[0], vertex[1], vertex[2], vertex[3], vertex[4], vertex[5])
    return xyzrgb

def array_to_pcd(data: np.ndArray):
    """Convert a numpy array to PCD format.
    data is a numpy array of shape (n, 7) where n is the number of vertices.
    where the fourth column is the class.
    Returns a string.
    """
    pcd = '# .PCD v.7 - Point Cloud Data file format\n'
    pcd += 'VERSION .7\n'
    pcd += 'FIELDS x y z rgb\n'
    pcd += 'SIZE 4 4 4 4\n'
    pcd += 'TYPE F F F F\n'
    pcd += 'COUNT 1 1 1 1\n'
    pcd += 'WIDTH %d\n' % data.shape[0]
    pcd += 'HEIGHT 1\n'
    pcd += 'VIEWPOINT 0 0 0 1 0 0 0\n'
    pcd += 'POINTS %d\n' % data.shape[0]
    pcd += 'DATA ascii\n'
    for vertex in data:
        pcd += '%f %f %f %d %d %d\n' % (vertex[0], vertex[1], vertex[2], vertex[3], vertex[4], vertex[5])
    return pcd
