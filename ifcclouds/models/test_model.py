import click
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dotenv import find_dotenv, load_dotenv

from ifcclouds.data.dataset import IfcCloudDs
from ifcclouds.models.dgcnn import DGCNN_semseg
from ifcclouds.visualization.visualize import plot_loss

@click.command()
@click.argument('checkpoint_path', type=click.Path(exists=True))
def main(checkpoint_path):
  """ Runs model testing """
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  dataset = IfcCloudDs(partition='test', num_points=4096)
  dataloader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=2)
  model = DGCNN_semseg(dataset.num_classes)
  model = nn.DataParallel(model)
  model.load_state_dict(torch.load(checkpoint_path))
  model.eval()
  total_loss = 0.0
  train_acc = 0.0
  for pointcloud, seg in dataloader:
      pointcloud = pointcloud.to(device)
      seg = seg.to(device)
      pred = model(pointcloud)
      pred = pred.permute(0, 2, 1).contiguous()

      loss = F.cross_entropy(pred.view(-1, dataset.num_classes), seg.view(-1,1).squeeze(), reduction='mean')
      loss_item = loss.item()
      total_loss = loss_item

      seg_pred = pred.max(dim=2)[1]
      correct = seg_pred.eq(seg.view(-1,1).squeeze()).cpu().sum()
      train_acc += correct.item() / (seg.size()[0] * seg.size()[1])

  print('Test Accuracy: %f' % (train_acc / len(dataloader)))
  print('Test Loss: %f' % (total_loss / len(dataloader)))

if __name__ == '__main__':
  load_dotenv(find_dotenv())
  main()