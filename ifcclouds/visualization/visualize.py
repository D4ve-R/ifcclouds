import os
import numpy as np
import matplotlib.pyplot as plt

def plot_pointcloud(pointcloud, label):
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  ax.scatter(pointcloud[:,0], pointcloud[:,1], pointcloud[:,2], c=label)
  plt.show()

def open_lidarview():
  os.system('open http://lidarview.com/')


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