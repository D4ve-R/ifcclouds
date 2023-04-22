import click
import numpy as np

from ifcclouds.utils import read_ply
from ifcclouds.visualization.visualize import plot_class_occurences

@click.command()
@click.argument('file', type=click.Path(exists=True))
def main(file):
    vertex_data = read_ply(file)
    vertices = np.zeros(len(vertex_data), dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('class', 'i4')])
    for i in range(len(vertex_data)):
        vertices[i] = (vertex_data[i][0], vertex_data[i][1], vertex_data[i][2], vertex_data[i][3])

    x = vertices['x']
    y = vertices['y']
    z = vertices['z']
    class_attr = vertices['class']
    plot_class_occurences(class_attr)
    

if __name__ == '__main__':
    main()