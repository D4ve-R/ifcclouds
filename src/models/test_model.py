import click
from torch.utils.data import DataLoader
from dotenv import find_dotenv, load_dotenv

from src.data.dataset import IfcCloudDs
from src.models.dgcnn import DGCNN

@click.command()
@click.argument('model', type=click.Path(exists=True))
def main(model):
  """ Runs model testing """
  dataset = IfcCloudDs(partition='test', num_points=4096, test_sample='1')
  dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
  model = DGCNN(num_classes=5, k=20)
  model.load_state_dict(torch.load(model))
  model.eval()
  for pointcloud, label in dataloader:
      print(pointcloud.shape)
      print(label.shape)
      pred = model(pointcloud)
      print(pred.shape)
      break

if __name__ == '__main__':
  load_dotenv(find_dotenv())
  main()