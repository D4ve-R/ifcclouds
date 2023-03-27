#/usr/bin/env python3
import sys
import os
import glob
import zlib

N = int(os.getenv('N', 1024 * 10))

def unzip(file_path):
    """Unzip a file and safe."""
    with open(file_path, 'rb') as f:
        data = zlib.decompress(f.read())
        return data

def unzip_dir(input_file, output_dir=None):
    """Unzip a directory and safe."""
    if output_dir is None:
        output_dir = os.path.dirname(input_file)
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
    with open(input_file, 'rb') as f:
        data = zlib.decompress(f.read())
        with open(os.path.join(output_dir, os.path.basename(input_file).split('.')[0]), 'wb') as out_file:
            out_file.write(data)

def main():
    if len(sys.argv) != 3:
        print('Usage: {} <input_dir> <output_dir>'.format(sys.argv[0]))
        sys.exit(1)
    input_dir = sys.argv[1]
    for input_file in glob.glob(input_dir + '/*.ifc'):
        output_dir = sys.argv[2]
        #output_dir = os.path.join(output_dir, os.path.basename(input_file).split('.')[0])
        output_file = os.path.basename(input_file).split('.')[0]
        print('Processing {}'.format(input_file))
        if(os.path.exists(os.path.join(output_dir, output_file+'.ply'))):
            print('Skipping {}'.format(input_file))
            continue

        os.system('N={} python3 convert.py {} {}'.format(N, input_file, output_dir))

if __name__ == '__main__':
    main()