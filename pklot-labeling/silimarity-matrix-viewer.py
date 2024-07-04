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


def show_similarity_matrix(file_name: str):
    """ Show the similarity matrix """

    image_name = os.path.basename(file_name)
    date = image_name.split('_')[0]

    with open(file_name, 'rb') as file:
        data = pickle.load(file)
        similarity_matrix = data['similarity_matrix']
        similarity_matrix = np.transpose(similarity_matrix) + similarity_matrix
        np.set_printoptions(precision=2)
        print(similarity_matrix)
        image_ids = data['image_ids']

        fig, ax = plt.subplots()
        plt.imshow(similarity_matrix, cmap='coolwarm', interpolation='nearest')
        plt.title(f'Date: {date}')
        # show heatmap with the similarity on the block
        for i in range(len(image_ids)):
            for j in range(len(image_ids)):
                plt.text(j, i, "{:.2f}".format(similarity_matrix[i, j]), ha='center', va='center', color='black')
        plt.xticks(range(len(image_ids)), image_ids, rotation=90)
        plt.yticks(range(len(image_ids)), image_ids)
        plt.xlabel('Image ID')
        plt.ylabel('Image ID')

        def on_move(event):
            if event.inaxes:
                print(f'data coords {event.xdata} {event.ydata},',
                    f'pixel coords {event.x} {event.y}')

        def on_click(event):
            print("clicked")
            # Get axis labels bounding boxes
            x_label = ax.xaxis.label.get_window_extent(renderer=fig.canvas.get_renderer())
            y_label = ax.yaxis.label.get_window_extent(renderer=fig.canvas.get_renderer())

            # Check if click is inside x-axis label bounding box
            if x_label.contains(event.x, event.y):
                print("Clicked on x-axis label")
            # Check if click is inside y-axis label bounding box
            elif y_label.contains(event.x, event.y):
                print("Clicked on y-axis label")

        # binding_id = plt.connect('motion_notify_event', on_move)
        # fig.canvas.mpl_connect('button_press_event', on_click)

        plt.show(block=True)

def test_mouse():
    from matplotlib.backend_bases import MouseButton

    t = np.arange(0.0, 1.0, 0.01)
    s = np.sin(2 * np.pi * t)
    fig, ax = plt.subplots()
    ax.plot(t, s)


    def on_move(event):
        if event.inaxes:
            print(f'data coords {event.xdata} {event.ydata},',
                f'pixel coords {event.x} {event.y}')


    def on_click(event):
        if event.button is MouseButton.LEFT:
            print('disconnecting callback')
            plt.disconnect(binding_id)


    binding_id = plt.connect('motion_notify_event', on_move)
    plt.connect('button_press_event', on_click)

    plt.show(block=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Show heatmap of similarity matrix')
    parser.add_argument('file', type=str, help='The matrix pickle file')
    args = parser.parse_args()

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    show_similarity_matrix(args.file)
    #test_mouse()