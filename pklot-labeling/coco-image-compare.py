# compare two images from a coco json file, extract the objects from the same location and compare them
#
# Usage: python coco-image-compare.py --coco_file coco.json --image_dir images
#

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


# try:
#     set_start_method('spawn')
# except RuntimeError:
#     pass


category_space_empty = 1
category_space_occupied = 2

def extract_patch(image, bbox):
    """ Extract the object from the image """
    x, y, w, h = bbox
    return image[y:y+int(h), x:x+int(w)]


def calc_image_hash(image):
    """ Calculate the phash of the image """
    return imagehash.phash(Image.fromarray(image))


def calculate_gray_histogram_similarity(image1, image2):
    """ Calculate the histogram similarity between two images """
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # calculate the histograms
    gray_hist1 = cv2.calcHist([gray1], [0], None, [256], [0, 256])
    gray_hist2 = cv2.calcHist([gray2], [0], None, [256], [0, 256])

    # calculate the correlation
    correlation = cv2.compareHist(gray_hist1, gray_hist2, cv2.HISTCMP_CORREL)
    return correlation

def calculate_color_histogram_similarity(image1, image2):
    """ Calculate the histogram similarity between two images """
    colors = ('b', 'g', 'r')
    hist1 = {}
    hist2 = {}

    for i, color in enumerate(colors):
        hist1[color] = cv2.calcHist([image1], [i], None, [256], [0, 256])
        hist2[color] = cv2.calcHist([image2], [i], None, [256], [0, 256])

    correlation = 0
    for color in colors:
        correlation += cv2.compareHist(hist1[color], hist2[color], cv2.HISTCMP_CORREL)

    return correlation / len(colors)


def calculate_iou(bbox1, bbox2):
    """ Calculate the Intersection over Union (IoU) between two bounding boxes """
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2

    xA = max(x1, x2)
    yA = max(y1, y2)
    xB = min(x1 + w1, x2 + w2)
    yB = min(y1 + h1, y2 + h2)

    interArea = max(0, xB - xA) * max(0, yB - yA)

    boxAArea = w1 * h1
    boxBArea = w2 * h2

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

# class Metric:
#     def __init__(self, image_hash):
#         self.image_hash = image_hash
        # self.color_hist = color_hist
        # self.gray_hist = gray_hist

class Patch:
    def __init__(self, ann_id:int, image_id:int, category_id:int, bbox, image):
        self.ann_id = ann_id # annotation id
        self.image_id = image_id
        self.category_id = category_id
        self.bbox = bbox
        self.image = image
        self.image_hash = calc_image_hash(image)

    # def calculate_metrics(self, image):
    #     image_hash = calc_image_hash(image)
    #     # color_hist = cv2.calcHist([image], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
    #     # gray_hist = cv2.calcHist([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)], [0], None, [256], [0, 256])
    #     return Metric(image_hash)

class ImageFeature:
    def __init__(self, image_id:int, image_path:str, patches:List[Patch]):
        self.image_id = image_id
        self.image_path = image_path
        self.patches = sorted(patches, key=lambda patch: (patch.bbox[0], patch.bbox[1]))

    def __sub__(self, other):
		# type: (ImageFeature) -> float
        if other is None:
            raise TypeError('Other feature must not be None.')
        return self.calculate_similarity(other)

    def __iter__(self):
        return iter(self.patches)
    
    def __len__(self):
        return len(self.patches)

    def calculate_similarity(self, other) -> float:
        similarity = 0.0

        overlapped = self.get_overlapping_patches(other)
        if len(overlapped) == 0:
            return 0

        # plt.figure(figsize=(20, 10))

        # rows = min(10, len(overlapped))

        # pos = 0

        identical = 0

        for patch1, patch2 in overlapped:
            diff = patch1.image_hash - patch2.image_hash
            # image1 = patch1.image
            # image2 = patch2.image
            # print(diff)

            # plt.subplot(rows, 6, pos + 1)
            # image1_rgb = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
            # plt.imshow(image1_rgb)
            # plt.axis('off')

            # plt.subplot(rows, 6, pos + 2)
            # image2_rgb = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
            # plt.imshow(image2_rgb)
            # plt.axis('off')

            # plt.subplot(rows, 6, pos + 3)
            # plt.title(f'Sim:  {diff}')
            # plt.axis('off')

            # pos += 3
            # if pos == 60:
            #     break

            # print("diff: ", diff)

            if diff < 14:
                identical += 1

            # similarity += diff #patch1.image_hash - patch2.image_hash

        # plt.show(block=True)
        #return similarity
        similarity = identical / len(overlapped)
        # print(f"Identical: {identical} / {len(overlapped)} = {similarity}")

        similarity = identical / max( len(self), len(other) )
        # print(f"similarity: {identical} / max({len(self)}, {len(other)}) = {similarity}")
        return similarity

    def get_overlapping_patches(self, other, iou_threshold=0.5):
        overlapping_patches = []

        for patch1 in self:
            for patch2 in other:
                iou = calculate_iou(patch1.bbox, patch2.bbox)
                if iou > iou_threshold:
                    overlapping_patches.append((patch1, patch2))

        # print("overlapped:", len(overlapping_patches))
        return overlapping_patches

class CocoComparer:

    def __init__(self, coco_file):
        self.coco = self.load_coco(coco_file)
        self.image_dir = os.path.dirname(coco_file)
        self.similarity_matrix = None
        self.lock = threading.Lock() 

    def load_coco(self, coco_file):
        """ Load the COCO json file """
        with open(coco_file, 'r') as f:
            coco = json.load(f)
        return coco

    def load_patches_by_category(self, image_file, image_id, category_id) -> ImageFeature:
        """ Load the patches from the image by category
        :param image_path: The path to the image
        :param image_id: The image id
        :param category_id: The category id
        :return: ImageFeature
        """

        image_path = os.path.join(self.image_dir, image_file)

        # print(image_path)
        # print("image id:", image_id)

        # Load the image
        image = cv2.imread(image_path)
        # rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        np_img = np.array(image)

        found = False

        objects:List[Patch] = []
        for annotation in self.coco['annotations']:
            if annotation['image_id'] == image_id:
                found = True
                if annotation['category_id'] == category_id:
                    # print('+', end="")
                    objects.append( Patch(annotation['id'], image_id, category_id, annotation['bbox'], extract_patch(np_img, annotation['bbox'])))
                # else:
                    # print('-',end="")
            elif found:
                # print('.',end="") # early break
                break

        # print('*')
        imageFeature = ImageFeature(image_id, image_file, objects)
        #pickle.dump(imageMetrics, open(f"{image_path}.pkl", "wb"))
        return imageFeature

    def get_date_from_image(self, image_path):
        """ Get the date from the image """
        image_name = os.path.basename(image_path)
        date = image_name.split('_')[0]
        return date

    def create_metrics(self) -> Dict[str, List[ImageFeature]]:
        """ Compare the images in the COCO json file """

        start_time = time.time()

        all_images = {}

        num = 0

        for image in tqdm(self.coco['images'], desc="Loading image patches"):
            num += 1
            img_date = self.get_date_from_image(image['file_name'])
            if img_date not in all_images:
                all_images[img_date] = []

            all_images[img_date].append(self.load_patches_by_category(
                image['file_name'], image['id'], category_space_occupied))
            # if num == 10:
            #     break

        end_time = time.time()
        time_elapsed = end_time - start_time
        print("Time elapsed:", time_elapsed, "seconds")
        return all_images

    def init_similarity_matrix(self, num_images):
        self.similarity_matrix = np.zeros((num_images, num_images))

    def update_matrix(self, i:int, j:int, similarity:float):
        with self.lock:
            self.similarity_matrix[i, j] = similarity

    def calculate_similarity(self, image1:ImageFeature, image2:ImageFeature, i, j):
        similarity = image1 - image2
        self.update_matrix(i, j, similarity)

    def create_similarity_matrix(self):
        """ Create the similarity matrix """
        all_images = self.create_metrics()

        num_cpus = os.cpu_count()
        print(f"Number of CPUs: {num_cpus}")

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_cpus) as executor:
            for date, images in all_images.items():
                print(f"Date: {date}, images: {len(images)}")
                num_images = len(images)
                self.init_similarity_matrix(num_images)

                for i in range(num_images):

                    futures = []

                    pbar = tqdm(total=(num_images-i-1), desc=f"{date} Image {images[i].image_id}")

                    for j in range(i+1, num_images):
                        futures.append(executor.submit(self.calculate_similarity, images[i], images[j], i, j))

                    for future in concurrent.futures.as_completed(futures):
                        pbar.update(1)

                    # close the progress bar
                    pbar.close()

                    # get the maximum similarity
                    max_similarity = np.max(self.similarity_matrix[i])
                    # get the index of the maximum similarity
                    max_index = np.argmax(self.similarity_matrix[i])
                    # print(f"{i} Max similarity: {max_similarity} with {all_images[i].image_id} <-> {all_images[max_index].image_id}")
                    # print(f"{i} image_path1: {os.path.join(self.image_dir, all_images[i].image_path)}")
                    # print(f"{i} image_path2: {os.path.join(self.image_dir, all_images[max_index].image_path)}")

            all_ids = [image.image_id for image in images]
            print(self.similarity_matrix)
            data = {
                'similarity_matrix': self.similarity_matrix,
                'image_ids': all_ids
            }
            pickle.dump(data, open(os.path.join(self.image_dir, f'{date}_similarity_matrix.pkl'), 'wb'))
        return

    def show_similarity_matrix(self):
        """ Show the similarity matrix """
        data = pickle.load(open(os.path.join(self.image_dir, 'similarity_matrix.pkl'), 'rb'))
        similarity_matrix = data['similarity_matrix']
        image_ids = data['image_ids']
        plt.figure(figsize=(10, 10))
        plt.imshow(similarity_matrix, cmap='hot', interpolation='nearest')
        plt.xticks(range(len(image_ids)), image_ids, rotation=90)
        plt.yticks(range(len(image_ids)), image_ids)
        plt.show(block=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compare two images from a COCO json file')
    parser.add_argument('coco_file', type=str, help='The COCO json file')
    args = parser.parse_args()

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    coco_comparer = CocoComparer(args.coco_file)
    #coco_comparer.show_similarity_matrix()
    coco_comparer.create_similarity_matrix()