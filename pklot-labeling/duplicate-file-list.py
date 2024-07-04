import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import glob
import json
import time
import pickle
from PIL import Image
import imagehash
import threading
from tqdm import tqdm
import sys
from multiprocessing import set_start_method
import concurrent.futures
from typing import List, Dict
from matplotlib.backend_bases import MouseButton


def list_duplicate_files(pickle_file: str, annotation_file: str, threshold: float):
    """ List duplicate files """

    print(f'Processing file: {pickle_file}')

    with open(annotation_file, 'r') as f:
        data = json.load(f)

    images = data['images']

    # create a dictionary of image_id to file_name
    image_id_to_file_name = {}
    for image in images:
        image_id_to_file_name[image['id']] = image['file_name']

    with open(pickle_file, 'rb') as file:
        data = pickle.load(file)
        similarity_matrix = data['similarity_matrix']
        np.set_printoptions(precision=2)
        # print(similarity_matrix.shape)
        # print(similarity_matrix)
        image_ids = data['image_ids']

    image_list1 = []
    image_list2 = []

    # find the duplicate images by similarity > threshold
    for i in range(len(image_ids)):
        for j in range(len(image_ids)):
            if similarity_matrix[i, j] > threshold and i != j:
                image_list1.append(image_id_to_file_name[image_ids[i]])
                image_list2.append(image_id_to_file_name[image_ids[j]])
                print(image_ids[i], image_id_to_file_name[image_ids[i]],
                                         image_ids[j], image_id_to_file_name[image_ids[j]],
                                         similarity_matrix[i, j])

    # unique the image_list1 and image_list2
    image_list1 = list(set(image_list1))
    image_list2 = list(set(image_list2))

    # sort image_list1 and image_list2
    image_list1.sort()
    image_list2.sort()

    # write image_list1 to a file called image_list1.txt
    with open('image_list1.txt', 'w') as f:
        for image in image_list1:
            f.write(f'{image}\n')

    # write image_list2 to a file called image_list2.txt
    with open('image_list2.txt', 'w') as f:
        for image in image_list2:
            f.write(f'{image}\n')

    # check if image_list1 and image_list2 has repeated images
    repeated = set(image_list1).intersection(set(image_list2))
    print('Repeated images:', len(repeated))
    print('Image List2 length:', len(image_list2))

    # remove the repeated images from image_list2
    image_list2 = [image for image in image_list2 if image not in repeated]

    print('To be removed:', len(image_list2))

    # write the image_list2 to a file called to_be_removed.txt
    with open('to_be_removed.txt', 'a') as f:
        for image in image_list2:
            f.write(f'{image}\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Load similarity matrix and list duplicate images')
    parser.add_argument('pickle', type=str, help='The matrix pickle file or directory containing the files')
    parser.add_argument('annotation', type=str, help='The coco json annotation file')
    parser.add_argument('--threshold', type=float, help='The threshold')
    args = parser.parse_args()

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    print(args)

    if os.path.isdir(args.pickle):
        files = glob.glob(os.path.join(args.pickle, '*.pkl'))
        for file in files:
            list_duplicate_files(file, args.annotation, args.threshold)
    else:
        list_duplicate_files(args.pickle, args.annotation, args.threshold)