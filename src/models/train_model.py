import click
import torch
from torch.utils.data import DataLoader
from dotenv import find_dotenv, load_dotenv

from src.data.dataset import IfcCloudDs
from src.models.dgcnn import DGCNN

@click.command()
@click.argument('model', default='dgcnn')
@click.argument('checkpoint_dir', default='models', type=click.Path(exists=True))
@click.argument('cuda', default=True, type=bool)
def main(model, checkpoints, cuda):
  """ Runs model training """
  device = torch.device("cuda" if cuda else "cpu")
  dataset = IfcCloudDs(partition='train', num_points=4096, test_sample='1')
  dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
  if model == 'dgcnn':
    model = DGCNN(num_classes=5, k=20).to(device)
  else:
    raise NotImplementedError('Model not implemented')

  model.load_state_dict(torch.load(model))
  model = nn.DataParallel(model)
  model.train()
  for pointcloud, label in dataloader:
    print(pointcloud.shape)
    print(label.shape)
    pred = model(pointcloud)
    print(pred.shape)
    break

if __name__ == '__main__':
  load_dotenv(find_dotenv())
  main()