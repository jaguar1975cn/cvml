""" rename pklot file:
from 
    2013-04-15_17_50_12_jpg.rf.fdd3b646077cd661ef4c6f46c6d6bbb6.jpg 
to 
    UFPR05-Sunny-2013-04-15_17_50_12.jpg

a target name list is required
"""

import os
import sys
import argparse
import glob
from tqdm import tqdm

def extract_date_from_target_name(image_name):
    # from UFPR05-Sunny-2013-04-15_17_50_12.jpg to 2013-04-15_17_50_12
    date = image_name[-23:-4]
    return date

def create_target_name_dictionary(target_name_list_file):
    # read the target name list
    with open(target_name_list_file, 'r') as f:
        target_name_list = f.readlines()

    mapper = { extract_date_from_target_name(x.strip()): x.strip() for x in target_name_list }
    return mapper

def extract_date_from_src_name(image_name):
    # from 2013-04-15_17_50_12_jpg.rf.fdd3b646077cd661ef4c6f46c6d6bbb6.jpg to 2013-04-15_17_50_12
    date = image_name[0:19]
    return date

def do_rename(image_dir, mapper):
    # read the file list
    file_list = glob.glob(os.path.join(image_dir, "*.jpg"))

    # rename the files
    for image_file in tqdm(file_list):
        image_name = os.path.basename(image_file)
        date = extract_date_from_src_name(image_name)
        target_name = mapper[date]
        os.rename(image_file, os.path.join(image_dir, target_name))

    print('Done')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Rename pklot file')
    parser.add_argument('image_dir', type=str, help='The directory containing the images')
    parser.add_argument('target_name_list_file', type=str, help='The target name list file')

    args = parser.parse_args()

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    mapper = create_target_name_dictionary(args.target_name_list_file)

    do_rename(args.image_dir, mapper)