#/usr/bin/env python3
import sys
import argparse
import json
import os
import numpy as np
from numpy.typing import ArrayLike as ndArray
import ifcopenshell
import ifcopenshell.geom
from tqdm import tqdm

from ifcclouds.utils import load_classes_from_json
from ifcclouds.data.dataset import default_classes

argparser = argparse.ArgumentParser(description='Converts a file from ifc format to ply')
argparser.add_argument('input', help='Input file')
argparser.add_argument('output', help='Output dir')
argparser.add_argument('-n', '--num_points', help='Average number of points per m^2', default=1000)
argparser.add_argument('-c', '--classes', help='Classes to extract', default=os.path.join(os.path.dirname(__file__), 'data', 'classes.json'))
argparser.add_argument('-f', '--format', help='Output format', default='ply')
argparser.add_argument('-v', '--verbose', help='Verbose output', action='store_true')
argparser.add_argument('-d', '--debug', help='Debug output', action='store_true')

def local_to_world(origin, transform, verts):
    return origin + np.matmul(transform.T, verts.T).T

def array_to_ply(data: ndArray):
    """Convert a numpy array to PLY format.
    data is a numpy array of shape (n, 4) where n is the number of vertices.
    where the fourth column is the class.
    Returns a string.
    """
    ply = 'ply\n'
    ply += 'format ascii 1.0\n'
    ply += 'element vertex %d\n' % data.shape[0]
    ply += 'property float x\n'
    ply += 'property float y\n'
    ply += 'property float z\n'
    ply += 'property int class\n'
    ply += 'end_header\n'
    for vertex in data:
        ply += '%f %f %f %d\n' % (vertex[0], vertex[1], vertex[2], vertex[3])
    return ply

def barycentric(N):
    """Generate N random barycentric coordinates. 
    Returns three numpy arrays of shape (N, 1).
    """
    u = np.random.rand(N, 1)
    v = np.random.rand(N, 1)

    lp = u + v > 1
    u[lp] = 1 - u[lp]
    v[lp] = 1 - v[lp]
    w = 1 - (u + v)
    return u, v, w

def triangle_areas(v1: ndArray, v2: ndArray, v3: ndArray):
    """Calculate the area of multiple triangle given its vertices.
    Parameters are numpy arrays of shape (n, 3) where n is the number of triangles.
    """
    return 0.5 * np.linalg.norm(np.cross(v2 - v1, v3 - v1), axis=1)

def gen_pointcloud(vertices: ndArray, faces: ndArray, num_points=1000):
    """Generate a point cloud from a mesh.
    Parameters are vertices and faces of a mesh as numpy array.
    Returns a numpy array of shape (num_points, 3) with dtype=float32.
    """
    v1 = vertices[faces[:, 0]]
    v2 = vertices[faces[:, 1]]
    v3 = vertices[faces[:, 2]]
    
    areas = triangle_areas(v1, v2, v3)
    totalArea = np.sum(areas)
    probs = areas / totalArea
    num_points = totalArea.astype(int) * num_points
    random_indices = np.random.choice(range(len(areas)), num_points, p=probs)
    v1 = v1[random_indices]
    v2 = v2[random_indices]
    v3 = v3[random_indices]
    u, v, w = barycentric(num_points)
    points = u * v1 + v * v2 + w * v3
    return points.astype(np.float32)

def process_ifc_file(args):
    ifc_file_path = args.input
    out_path = args.output
    num_points = args.num_points
    verbose = args.verbose
    debug = args.debug
    class_path = args.classes

    if verbose: print('Processing %s' % ifc_file_path)
    ifc_file_name = os.path.basename(ifc_file_path).split('.')[0]
    ifc_file = ifcopenshell.open(ifc_file_path)
    settings = ifcopenshell.geom.settings()
    class_names = load_classes_from_json(class_path, verbose)
    if class_names is None:
        class_names = default_classes
    classes = {ifc_class: None for ifc_class in class_names}
    for ifc_class in tqdm(class_names):
        if verbose: print('Processing %s' % ifc_class)
        entities = 0
        try:
            for ifc_entity in ifc_file.by_type(ifc_class):
                entities += 1
                if verbose: print('Processing %s Entity No. %d' % (ifc_class, entities))
                shape = None
                try:
                    shape = ifcopenshell.geom.create_shape(settings, ifc_entity)

                    matrix = shape.transformation.matrix.data
                    matrix = np.array(matrix).reshape(4, 3)
                    origin = matrix[-1]
                    transform = matrix[:-1]
                    verts = shape.geometry.verts
                    faces = shape.geometry.faces

                    verts = np.array([[verts[i], verts[i + 1], verts[i + 2]] for i in range(0, len(verts), 3)])
                    faces = np.array([[faces[i], faces[i + 1], faces[i + 2]] for i in range(0, len(faces), 3)])
                    verts = local_to_world(origin, transform, verts)

                    points = gen_pointcloud(verts, faces, num_points)
                    if classes[ifc_class] is None: classes[ifc_class] = []
                    classes[ifc_class].append(points)

                except Exception as e:
                    if debug: print(e)
                    print('Error creating shape for %s Entity No. %d' % (ifc_class, entities))
                    continue
        except Exception as e:
            if debug: print(e)
            print('Error processing %s' % ifc_class)
            continue
          
        if verbose: print('Processed %d entities of type %s' % (entities, ifc_class))

    all_points = []
    for ifc_class in class_names:
        if classes[ifc_class] is not None:
            points = np.concatenate(classes[ifc_class])
            labels = np.full((points.shape[0], 1), class_names.index(ifc_class))
            labeled_points = np.concatenate((points, labels), axis=1)
            all_points.append(labeled_points)

    with open(os.path.join(out_path, ifc_file_name+'.ply'), 'w') as out_file:
        out_file.write(array_to_ply(np.concatenate(all_points)))

def process_ifc(ifc_file_path, out_path, num_points=4096):
    ifc_file_name = os.path.basename(ifc_file_path).split('.')[0]
    ifc_file = ifcopenshell.open(ifc_file_path)
    settings = ifcopenshell.geom.settings()
    class_names = load_classes_from_json(os.path.join(os.path.dirname(__file__), 'data', 'classes.json'))
    if class_names is None:
        class_names = default_classes
    classes = {ifc_class: None for ifc_class in class_names}
    for ifc_class in class_names:
        try:
            for ifc_entity in ifc_file.by_type(ifc_class):
                shape = None
                shape = ifcopenshell.geom.create_shape(settings, ifc_entity)

                matrix = shape.transformation.matrix.data
                matrix = np.array(matrix).reshape(4, 3)
                origin = matrix[-1]
                transform = matrix[:-1]
                verts = shape.geometry.verts
                faces = shape.geometry.faces
    
                verts = np.array([[verts[i], verts[i + 1], verts[i + 2]] for i in range(0, len(verts), 3)])
                faces = np.array([[faces[i], faces[i + 1], faces[i + 2]] for i in range(0, len(faces), 3)])
                verts = local_to_world(origin, transform, verts)
    
                points = gen_pointcloud(verts, faces, num_points)
                if classes[ifc_class] is None: classes[ifc_class] = []
                classes[ifc_class].append(points)
        except Exception as e:
            continue

    all_points = []
    for ifc_class in class_names:
        if classes[ifc_class] is not None:
            points = np.concatenate(classes[ifc_class])
            labels = np.full((points.shape[0], 1), class_names.index(ifc_class))
            labeled_points = np.concatenate((points, labels), axis=1)
            all_points.append(labeled_points)

    with open(os.path.join(out_path, ifc_file_name+'.ply'), 'w') as out_file:
        out_file.write(array_to_ply(np.concatenate(all_points)))
            
def main(argv = sys.argv[1:]):
    args = argparser.parse_args(argv)
    if args.debug:
        print(args)

    if args.verbose:
        print('Converting %s to %s as %s' % (args.input, args.output, args.format))

    if args.format == 'ply':
        if args.verbose: print('Converting %s to %s' % (args.input, args.output))
        process_ifc_file(args)
    else:
        print('Unknown format: %s' % args.format)

if __name__ == '__main__':
    main()
