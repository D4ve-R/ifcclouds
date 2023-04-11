import os
import glob
import json
import numpy as np
import random
import torch
from torch.utils.data import Dataset

DATADIR=os.path.join('data', 'processed')

default_classes = [
    "IfcBeam", 
    "IfcDoor", 
    "IfcFurniture", 
    "IfcFurnishingElement",
    "IfcLamp", 
    "IfcOutlet", 
    "IfcPipeSegment", 
    "IfcRailing", 
    "IfcSlab", 
    "IfcStair", 
    "IfcWall", 
    "IfcWindow",
    "IfcRoof",
    "IfcRamp",  
]

def one_hot_encode(label, num_classes):
  """
  One hot encodes the label.
  """
  label_one_hot = np.zeros((label.shape[0], num_classes))
  for idx in range(label.shape[0]):
    label_one_hot[idx, label[idx]] = 1
  return torch.from_numpy(label_one_hot.astype(np.float32))

def read_ply(file, partition, num_points):
  """
  Reads a ply file and returns a numpy array of shape (n, 3) and an array of shape(n,1).
  """
  #print('Preprocessing data...')
  lines = []
  with open(file, 'r') as f:
    lines = f.readlines()
  
    # remove header
    lines = lines[8:]

    indices = list(range(len(lines)))
    sampled_indices = random.sample(indices, num_points)
    if partition == 'train':
      np.random.shuffle(sampled_indices)

    lines = [lines[i] for i in sampled_indices]

    data = np.array([line.split()[:-1] for line in lines], dtype=np.float32)
    label = np.array([line.split()[-1] for line in lines], dtype=np.int32)

    return data, label

def load_ifccloud_ds(partition, test_sample):
  """
  Loads the ifccloud dataset. Returns a numpy array of shape (n, 4096, 3) and an array of shape(n,4096,1).
  """
  files = []
  if partition == 'train':
    files = glob.glob(os.path.join(DATADIR, '*.ply'))
  elif partition == 'test':
    raise NotImplementedError('Test partition not implemented yet')
    files = glob.glob(os.path.join(DATADIR, 'test', '*.ply'))
  else:
    raise ValueError('Partition must be train or test')
  
  return files

def load_classes_from_json(json_file_path, verbose=False):
    if verbose: print('Loading classes from %s' % json_file_path)
    try:
        with open(json_file_path) as json_file:
            return json.load(json_file)
    except Exception as e:
        if verbose: print(e)
        print('Error loading classes from %s, return default' % json_file_path)
        return default_classes

class IfcCloudDs(Dataset):
  def __init__(self, cache, partition='train', num_points=4096, test_sample='1'):
    self.cache = cache
    self.files = []
    for file in load_ifccloud_ds(partition, test_sample):
      self.files.append(file)
    self.num_points = num_points
    self.partition = partition
    self.classes = load_classes_from_json(os.path.join(os.path.dirname(__file__), 'classes.json'))
    self.num_classes = len(self.classes)

  def __getitem__(self, item):
    if item not in self.cache:
      self.cache[item] = read_ply(self.files[item], self.partition, self.num_points)
    
    pointcloud, seg = self.cache[item]

    # one hot encode the labels
    label = one_hot_encode(seg, self.num_classes)
    
    #indices = list(range(pointcloud.shape[0]))
    #sampled_indices = random.sample(indices, self.num_points)

    #if self.partition == 'train':
    #  np.random.shuffle(sampled_indices)
    #  pointcloud = pointcloud[sampled_indices]
    #  label = label[sampled_indices]
    seg = torch.LongTensor(seg)

    return pointcloud, label, seg

  def __len__(self):
    return len(self.files)