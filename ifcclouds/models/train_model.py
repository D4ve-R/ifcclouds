import os
import click
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import optim
from dotenv import find_dotenv, load_dotenv
from tqdm import tqdm
from multiprocessing import Manager
from datetime import datetime

from ifcclouds.data.dataset import IfcCloudDs
from ifcclouds.models.dgcnn import DGCNN_partseg
from ifcclouds.visualization.visualize import plot_pointcloud, open_lidarview

def save_model(model, checkpoint_dir, epoch):
  """ Saves the model to the checkpoint directory """
  timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
  torch.save(model.state_dict(), os.path.join(checkpoint_dir, timestamp + '_model_%d.pth' % epoch))

def load_model(model, checkpoint_path):
  """ Loads the model from the checkpoint directory """
  model.load_state_dict(torch.load(checkpoint_path))

@click.command()
@click.argument('model_type', default='dgcnn')
@click.argument('checkpoint_path', default='models/')
@click.option('--epochs', default=100, type=int)
@click.option('--lr', default=0.001, type=float)
@click.option('--momentum', default=0.9, type=float)
@click.option('--use_sgd', default=False, type=bool)
@click.option('--cuda', default=True, type=bool)
def main(model_type, checkpoint_path, epochs, lr, momentum, use_sgd, cuda):
  """ Runs model training """
  device = torch.device("cuda" if cuda and torch.cuda.is_available() else "cpu")

  manager = Manager()
  cache = manager.dict()
  dataset = IfcCloudDs(cache, partition='train', num_points=4096, test_sample='1')
  dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=2)
  model = None
  if model_type == 'dgcnn':
    model = DGCNN_partseg(dataset.num_classes).to(device)
  elif model_type == 'pointnet':
    raise NotImplementedError('Model not implemented')
  else:
    raise NotImplementedError('Model not implemented')
  
  #print(str(model))

  if use_sgd:
    opt = optim.SGD(model.parameters(), lr=lr*100, momentum=momentum, weight_decay=1e-4)
  else:
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

  if checkpoint_path != 'models/':
    load_model(model, checkpoint_path)
  
  model = nn.DataParallel(model)

  for epoch in tqdm(range(epochs)):
    model.train()
    for pointcloud, label, seg in dataloader:
      pointcloud = pointcloud.to(device)
      label = label.to(device)
      seg = seg.to(device)

      label = torch.ones((2, 13, 1), device=device)

      #print(pointcloud.shape)
      #print(label[0][:5])
      #print(seg.shape)
      
      #plot_pointcloud(pointcloud[0], seg[0])

      opt.zero_grad()
      pred = model(pointcloud, label)

      loss = F.cross_entropy(pred, seg, reduction='mean')
      loss.backward()
      opt.step()
      print('Loss: %f' % loss.item())

    save_model(model, os.path.dirname(checkpoint_path), epoch)

if __name__ == '__main__':
  load_dotenv(find_dotenv())
  main()