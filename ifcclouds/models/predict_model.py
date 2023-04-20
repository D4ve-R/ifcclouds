import click
import torch
import torch.nn as nn

from ifcclouds.models.dgcnn import DGCNN_semseg
from ifcclouds.data.dataset import default_classes

NUM_CLASSES = len(default_classes)

# save pointcloud and seg to ply file
def save_pointcloud(pointcloud, seg):
    ply = 'ply\n'
    ply += 'format ascii 1.0\n'
    ply += 'element vertex %d\n' % pointcloud.shape[0]
    ply += 'property float x\n'
    ply += 'property float y\n'
    ply += 'property float z\n'
    ply += 'property int class\n'
    ply += 'end_header\n'
    for i in range(pointcloud.shape[0]):
        ply += '%f %f %f %d \n' % (pointcloud[i, 0], pointcloud[i, 1], pointcloud[i, 2], seg[i])
    return ply


@click.command()
@click.argument('checkpoint_path', type=click.Path(exists=True))
def main(checkpoint_path):
  """ Runs model inference """
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = DGCNN_semseg(NUM_CLASSES).to(device)
  model = nn.DataParallel(model)
  model.load_state_dict(torch.load(checkpoint_path))
  model.eval()
  # TODO: segment a pointcloud
  pointcloud = None
  pred = model(pointcloud)
  print(pred)

if __name__ == '__main__':
    main()
