# -*- coding: utf-8 -*-
import os
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import glob
import multiprocessing

    
def convert_file(input_file, output_dir):
    os.system('python3 -m ifcclouds.convert {} {}'.format(input_file, output_dir))


@click.command()
@click.argument('input_dir', default='data/raw', type=click.Path(exists=True))
@click.argument('output_dir', default='data/processed', type=click.Path())
def main(input_dir, output_dir):
    """ Runs data processing scripts to turn raw data from (data/raw) into
        cleaned data ready to be analyzed (saved in data/processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    pool = multiprocessing.Pool()
    for input_file in glob.glob(os.path.join(input_dir, '*.ifc')):
        output_file = os.path.basename(input_file).split('.')[0]
        
        if(os.path.exists(os.path.join(output_dir, output_file+'.ply'))):
            logger.info('Skipping {}'.format(input_file))
            continue

        logger.info('Processing {}'.format(input_file))
        #os.system('python3 -m ifcclouds.convert {} {}'.format(input_file, output_dir))
        pool.apply_async(convert_file, args=(input_file, output_dir))
    
    pool.close()
    pool.join()


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
