import click
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dotenv import find_dotenv, load_dotenv

from ifcclouds.data.dataset import IfcCloudDs
from ifcclouds.models.dgcnn import DGCNN_partseg

@click.command()
@click.argument('model', default='dgcnn')
@click.argument('checkpoint_dir', default='')
@click.argument('cuda', default=False, type=bool)
def main(model, checkpoint_dir, cuda):
  """ Runs model training """
  device = torch.device("cuda" if cuda else "cpu")
  dataset = IfcCloudDs(partition='train', num_points=4096, test_sample='1')
  dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)
  if model == 'dgcnn':
    model = DGCNN_partseg(dataset.num_classes).to(device)
  else:
    raise NotImplementedError('Model not implemented')

  #model.load_state_dict(torch.load(model))
  if checkpoint_dir != '':
    model.load_state_dict(torch.load(checkpoint_dir))
  model = nn.DataParallel(model)
  model.train()
  for pointcloud, label in dataloader:
    print(pointcloud.shape)
    print(label.shape)
    pred = model(pointcloud, label)
    print(pred.shape)

  #model._save_to_state_dict(checkpoint_dir)

if __name__ == '__main__':
  load_dotenv(find_dotenv())
  main()