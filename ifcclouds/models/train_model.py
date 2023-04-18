import os
import click
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import optim
from dotenv import find_dotenv, load_dotenv
from tqdm import tqdm
from datetime import datetime

from ifcclouds.data.dataset import IfcCloudDs
from ifcclouds.models.dgcnn import DGCNN_semseg
from ifcclouds.visualization.visualize import plot_loss

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

  dataset = IfcCloudDs(partition='train', num_points=4096)
  dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=2)
  model = None
  if model_type == 'dgcnn':
    model = DGCNN_semseg(dataset.num_classes).to(device)
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

  total_loss = []

  for epoch in tqdm(range(epochs)):
    model.train()
    for pointcloud, seg in dataloader:
      pointcloud = pointcloud.to(device)
      seg = seg.to(device)

      opt.zero_grad()
      pred = model(pointcloud)
      pred = pred.permute(0, 2, 1).contiguous()

      loss = F.cross_entropy(pred.view(-1, 13), seg.view(-1,1).squeeze(), reduction='mean')
      loss.backward()
      opt.step()
      loss_item = loss.item()
      print('Loss: %f' % loss_item)
      total_loss.append(loss_item)

      seg_pred = pred.max(dim=2)[1]
      
    save_model(model, os.path.dirname(checkpoint_path), epoch)

  plot_loss(total_loss)

if __name__ == '__main__':
  load_dotenv(find_dotenv())
  main()