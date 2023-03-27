import os
import glob
import numpy as np
from torch.utils.data import Dataset

DATADIR=os.path.join('data', 'processed')

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
    files = glob.glob(os.path.join(DATADIR,'train', '*.ply'))
  elif partition == 'test':
    files = glob.glob(os.path.join(DATADIR, 'test', '*.ply'))
  else:
    raise ValueError('Partition must be train or test')
  points = []
  labels = []
  for file in files:
    data, label = read_ply(file)
    points.append(data)
    labels.append(label)
  return np.array(points), np.array(labels)

def load_classes_from_json(json_file_path, verbose=False):
  if verbose: print('Loading classes from %s' % json_file_path)
  with open(json_file_path) as json_file:
    return json.load(json_file)

class IfcCloudDs(Dataset):
  def __init__(self, partition='train', num_points=4096, test_sample='1'):
    self.points, self.label = load_ifccloud_ds(partition, test_sample)
    self.num_points = num_points
    self.partition = partition
    self.classes = load_classes_from_json(os.path.join('..', 'classes.json'))
    self.num_classes = len(self.classes)

  def __getitem__(self, item):
    pointcloud = self.points[item][:self.num_points]
    label = self.label[item][:self.num_points]
    if self.partition == 'train':
      indices = list(range(pointcloud.shape[0]))
      np.random.shuffle(indices)
      pointcloud = pointcloud[indices]
      label = label[indices]
    #label = torch.LongTensor(label)
    label = torch.tensor(label)
    return pointcloud, label

  def __len__(self):
    return self.points.shape[0]