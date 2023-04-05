import os
import matplotlib.pyplot as plt

def plot_pointcloud(pointcloud, label):
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  ax.scatter(pointcloud[:,0], pointcloud[:,1], pointcloud[:,2], c=label)
  plt.show()

def open_lidarview():
  os.system('open http://lidarview.com/')

