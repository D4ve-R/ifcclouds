import os
import glob
import numpy as np
import random
import torch
from torch.utils.data import Dataset
from multiprocessing import Manager

from ifcclouds.utils import load_classes_from_json, read_ply

DATADIR=os.path.join('data', 'processed')

class IfcCloudDs(Dataset):
  default_classes = [
    "IfcBeam", "IfcColumn", "IfcCovering", "IfcDoor", "IfcFurniture", "IfcPlate", "IfcRailing", "IfcRamp", "IfcRoof", "IfcStair", "IfcSlab", "IfcWall", "IfcWindow"
  ]

  def __init__(self, partition='train', num_points=4096):
    self.cache = Manager().dict()
    self.files = glob.glob(os.path.join(DATADIR, '*.ply'))
    num_files = len(self.files)
    num_train = int(num_files * 0.8)
    if partition == 'train':
      self.files = self.files[:num_train]
    elif partition == 'test':
      self.files = self.files[num_train:]
    else:
      raise ValueError('Partition must be train or test')
    
    self.num_points = num_points
    self.partition = partition
    self.classes = load_classes_from_json(os.path.join(os.path.dirname(__file__), 'classes.json'))
    if self.classes is None:
      self.classes = self.__class__.default_classes
    self.num_classes = len(self.classes)

  def __getitem__(self, item):
    if item not in self.cache:
      self.cache[item] = read_ply(self.files[item])
    points = self.cache[item]
    sampled_points = random.sample(list(range(len(points))), self.num_points)
    if self.partition == 'train':
      np.random.shuffle(sampled_points)

    points = [points[i] for i in sampled_points]
    data = [[pset[0], pset[1], pset[2]]  for pset in points]
    seg = [pset[3] for pset in points]
    #data = [list(map(float,line.split()[:-1])) for line in points]
    #seg = [int(line.split()[-1]) for line in points]
    return torch.tensor(data, dtype=torch.float32), torch.LongTensor(seg)

  def __len__(self):
    num_files = len(self.files)
    num_train = int(num_files * 0.8)
    if self.partition == 'train':
      return num_train
    else:
      return num_files - num_train