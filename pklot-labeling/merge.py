# A utility to view the COCO dataset
import argparse
import os
import sys
import json
#import matplotlib.pyplot as plt

if __name__ == '__main__':

    parser = argparse.ArgumentParser('PkLot annotation merge tool')
    parser.add_argument('--original-annotation', default="_annotations.coco.json", type=str)
    parser.add_argument('--enhanced-annotation', default="an.json", type=str)
    parser.add_argument('--output-annotation', default="output.json", type=str)

    args = parser.parse_args()

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    original_annotation_file = args.original_annotation
    enhanced_annotation_file = args.enhanced_annotation

    with open(original_annotation_file, 'r') as f:
        original_annotations = json.load(f)

    with open(enhanced_annotation_file, 'r') as f:
        enhanced_annotation = json.load(f)

    all_files_in_enhanced = set([x['file_name'] for x in enhanced_annotation['images']])

    # generate a map from image file name to image id in enhanced annotation
    enhanced_image_id_map = { x['file_name']: x['id'] for x in enhanced_annotation['images']}


    # now enumate all files in original annotations, if the image is in the enhanced annotation,
    # then we will remove its annotations from the original annotation, then insert the enhanced annotations
    # into the original annotation, remember to replace the anntations' image_id with the original image_id

    new_annotations = []

    # we will add annotions to the new_annotations, if the image is in the enhanced annotation, we will add the
    # enhanced annotations instead of the original annotations

    c = 0

    for image in original_annotations['images']:
        if image['file_name'] in all_files_in_enhanced:

            # we will add the enhanced annotations instead of the original annotations
            image_id = enhanced_image_id_map[image['file_name']]

            # print the image id and path, and a replaced icon (v)
            print(f"{image['id']}\t{image['file_name']}\t(v)", image_id, "->", image['id'])

            found = False

            for annotation in enhanced_annotation['annotations']:
                annotation = annotation.copy()
                if annotation['image_id'] == image_id:
                    annotation['image_id'] = image['id']
                    new_annotations.append(annotation)
                    found = True
                elif found:
                    break

        else:
            # we will add the original annotations

            # print the image id and path, and a replaced icon (x)
            print(f"{image['id']}\t{image['file_name']}\t(x)")

            found = False

            for annotation in original_annotations['annotations']:
                annotation = annotation.copy()
                if annotation['image_id'] == image['id']:
                    annotation['image_id'] = image['id']
                    new_annotations.append(annotation)
                    found = True
                elif found:
                    break

    # we need to reorder the new annotations from 0 to n
    for i, annotation in enumerate(new_annotations):
        # print i without newline
        if i< 100:
            print(i, end=' ', flush=True)
        annotation['id'] = i

    original_annotations['annotations'] = new_annotations

    # write the new json to a file
    with open(args.output_annotation, 'w') as f:
        json.dump(original_annotations, f, indent=4)