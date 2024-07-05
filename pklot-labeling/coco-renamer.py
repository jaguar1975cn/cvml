""" rename coco image files in the annotation:
from
    2013-04-15_17_50_12_jpg.rf.fdd3b646077cd661ef4c6f46c6d6bbb6.jpg 
to
    UFPR05-Sunny-2013-04-15_17_50_12.jpg

a target name list is required
"""

import os
import sys
import json
import argparse
import glob
from tqdm import tqdm
from renamer import extract_date_from_target_name, create_target_name_dictionary, extract_date_from_src_name


def do_rename(coco_file, mapper, output):
    # read the file list
    with open(coco_file, 'r') as f:
        data = json.load(f)

    # rename the files
    for image in tqdm(data['images']):
        date = extract_date_from_src_name(image['file_name'])
        target_name = mapper[date]
        image['file_name'] = target_name

    with open(output, 'w') as f:
        json.dump(data, f, indent=4)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Rename files in coco annotation')
    parser.add_argument('coco_file', type=str, help='The coco annotation file')
    parser.add_argument('target_name_list_file', type=str, help='The target name list file')
    parser.add_argument('output', type=str, help='The output file')

    args = parser.parse_args()

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    mapper = create_target_name_dictionary(args.target_name_list_file)

    do_rename(args.coco_file, mapper, args.output)