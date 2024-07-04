import numpy as np
import sys
import argparse
import pickle
import json

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Print similarity matrix')
    parser.add_argument('pickle', type=str, help='The matrix pickle file or directory containing the files')
    parser.add_argument('annotation', type=str, help='The json annotation file')
    args = parser.parse_args()

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    with open(args.annotation, 'r') as file:
        annotation = json.load(file)

    image_id_to_file_name = {}
    for image in annotation['images']:
        image_id_to_file_name[image['id']] = image['file_name']    

    with open(args.pickle, 'rb') as file:
        data = pickle.load(file)

    similarity_matrix = data['similarity_matrix']
    image_ids = data['image_ids']

    print(image_ids)
    for iid in image_ids:
        print(f"{iid} {image_id_to_file_name[iid]}")
    print(similarity_matrix)
        
