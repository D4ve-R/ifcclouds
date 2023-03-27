# -*- coding: utf-8 -*-
import os
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import glob
import zlib

def unzip(file_path):
    """Unzip a file and safe."""
    with open(file_path, 'rb') as f:
        data = zlib.decompress(f.read())
        return data

@click.command()
@click.argument('input_dir', default='data/raw', type=click.Path(exists=True))
@click.argument('output_dir', default='data/processed', type=click.Path())
def main(input_dir, output_dir):
    """ Runs data processing scripts to turn raw data from (data/raw) into
        cleaned data ready to be analyzed (saved in data/processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    for input_file in glob.glob(input_dir + '/*.ifc'):
        output_file = os.path.basename(input_file).split('.')[0]
        print('Processing {}'.format(input_file))
        if(os.path.exists(os.path.join(output_dir, output_file+'.ply'))):
            print('Skipping {}'.format(input_file))
            continue
        os.system('python3 convert.py {} {}'.format(input_file, output_dir))


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
