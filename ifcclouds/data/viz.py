import click

from ifcclouds.visualization.visualize import viz_dataset, open_in_browser

@click.command()
@click.option('-f', '--file', default='', help='visualize file in borwser')
@click.option('-d', '--dataset', help='visualize all samples from a dataset', type=bool)
@click.option('-ns', '--num_samples', default=-1, help='Number of samples to visualize')
@click.option('-p', '--partition', default='train', help='Partition to visualize')
@click.option('-np', '--num_points', default=4096, help='Number of points per sample')
def main(file, dataset, num_samples, partition, num_points):
  if dataset:
    viz_dataset(num_samples, partition, num_points)
  else:
    file = file.split('/')[-1]
    open_in_browser(file)


if __name__ == '__main__':
  main()