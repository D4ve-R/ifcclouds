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

def read_ply(filename):
  """
  Reads a ply file and returns a numpy array of shape (n, 3) and an array of shape(n,1).
  ply has format float float float int
  """
  with open(filename, 'r') as f:
    lines = f.readlines()
  for i, line in enumerate(lines):
    if line.startswith('end_header'):
      break
  data = np.array([line.split()[:-1] for line in lines[i+1:]], dtype=np.float32)
  label = np.array([line.split()[-1] for line in lines[i+1:]], dtype=np.int32)
  return data, label

def load_ifccloud_ds(partition, test_sample):
  """
  Loads the ifccloud dataset. Returns a numpy array of shape (n, 4096, 3) and an array of shape(n,4096,1).
  """
  if partition == 'train':
    files = glob.glob(os.path.join(DATADIR, '*.ply'))
  elif partition == 'test':
    raise NotImplementedError('Test partition not implemented yet')
    files = glob.glob(os.path.join(DATADIR, 'test', '*.ply'))
  else:
    raise ValueError('Partition must be train or test')
  points = []
  labels = []
  for file in files:
    data, label = read_ply(file)
    points.append(data)
    labels.append(label)
  return points, labels

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
  def __init__(self, partition='train', num_points=4096, test_sample='1'):
    self.points, self.label = load_ifccloud_ds(partition, test_sample)
    self.num_points = num_points
    self.partition = partition
    self.classes = load_classes_from_json('classes.json')
    self.num_classes = len(self.classes)

  def __getitem__(self, item):
    pointcloud = self.points[item]
    label = self.label[item]
    indices = list(range(pointcloud.shape[0]))
    sampled_indices = random.sample(indices, self.num_points)

    if self.partition == 'train':
      np.random.shuffle(sampled_indices)
      pointcloud = pointcloud[sampled_indices]
      label = label[sampled_indices]
    #label = torch.LongTensor(label)
    label = torch.tensor(label)
    return pointcloud, label

  def __len__(self):
    return len(self.points)