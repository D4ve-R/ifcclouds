import os
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from ifcclouds.data.dataset import IfcCloudDs

def open_in_browser(filename, host='localhost', port=8080):
  url = 'http://{}:{}/viz?filename={}'.format(host, port, filename)
  os.system('open {}'.format(url))

def plot_pointcloud(pointcloud, label):
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  ax.scatter(pointcloud[:,0], pointcloud[:,1], pointcloud[:,2], c=label)
  plt.show()

def classToRGB(num_classes):
  """ Returns a color map for the number of classes """
  cmap = plt.get_cmap('gist_rainbow')
  return cmap(np.linspace(0, 1, num_classes))
  
def plot_loss(loss):
  """ Plots the loss """
  plt.plot(loss)
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.show()

def viz_dataset(num_samples=-1, partition='train', num_points=4096):
  print('Visualizing {} samples from {} partition'.format(num_samples if num_samples > 0 else 'all', partition))
  dataset = IfcCloudDs(partition='train', num_points=num_points)
  print('Dataset size: {}'.format(len(dataset)))
  dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
  print('Dataloader size: {}'.format(len(dataloader)))
  sample_count = 0
  for pointcloud, seg in dataloader:
    plot_pointcloud(pointcloud[0].numpy(), seg[0].numpy())
    sample_count += 1
    if num_samples > 0 and sample_count >= num_samples:
      break