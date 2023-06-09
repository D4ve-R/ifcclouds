import os
import click
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from torch import optim
from dotenv import find_dotenv, load_dotenv
from tqdm import tqdm
from datetime import datetime

from ifcclouds.data.dataset import IfcCloudDs
from ifcclouds.models.dgcnn import DGCNN_semseg
from ifcclouds.visualization.visualize import plot_loss

@click.command()
@click.argument('model_type', default='dgcnn')
@click.argument('checkpoint_path', default='models/')
@click.option('--epochs', default=100, type=int)
@click.option('--lr', default=0.001, type=float)
@click.option('--momentum', default=0.9, type=float)
@click.option('--use_sgd', default=False, type=bool)
@click.option('--step', default=False, type=bool)
@click.option('--cuda', default=True, type=bool)
def main(model_type, checkpoint_path, epochs, lr, momentum, use_sgd, step, cuda):
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

  scheduler = StepLR(opt, step_size=20, gamma=0.5, last_epoch=-1) if step else CosineAnnealingLR(opt, epochs, eta_min=1e-3)
  
  model = nn.DataParallel(model)

  if checkpoint_path != 'models/':
    model.load_state_dict(torch.load(checkpoint_path))

  total_loss = []
  train_acc = 0.0

  for epoch in tqdm(range(epochs)):
    model.train()
    for pointcloud, seg in dataloader:
      pointcloud = pointcloud.to(device)
      seg = seg.to(device)

      opt.zero_grad()
      pred = model(pointcloud)
      pred = pred.permute(0, 2, 1).contiguous()

      loss = F.cross_entropy(pred.view(-1, dataset.num_classes), seg.view(-1,1).squeeze(), reduction='mean')
      loss.backward()
      opt.step()
      loss_item = loss.item()
      total_loss.append(loss_item)

      seg_pred = pred.max(dim=2)[1]
      correct = seg_pred.eq(seg).sum().item()
      train_acc += correct / (seg.size()[0] * seg.size()[1])
    if step:
      if opt.param_groups[0]['lr'] > 1e-5:
        scheduler.step()
      if opt.param_groups[0]['lr'] < 1e-5:
        for param_group in opt.param_groups:
          param_group['lr'] = 1e-5
    else:
      scheduler.step()
      
  timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
  checkpoint_file = timestamp + '_%s.pth' % model_type
  torch.save(model.state_dict(keep_vars=True), os.path.join(os.path.dirname(checkpoint_path), checkpoint_file))
  print('Train Accuracy: %f' % (train_acc / (len(dataloader) * epochs)))
  plot_loss(total_loss)

  os.system('python3 -m ifcclouds.models.test_model %s' % os.path.join(os.path.dirname(checkpoint_path), checkpoint_file))

if __name__ == '__main__':
  load_dotenv(find_dotenv())
  main()