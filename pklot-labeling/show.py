import sys
import os
from torchvision.datasets import CocoDetection
from torchvision.transforms import transforms
import matplotlib.pyplot as plt
import torchvision.transforms as T
import argparse

def show(img, targets):
    # Plot the image with the bounding boxes
    fig, ax = plt.subplots()

    transform = T.ToPILImage()

    ax.imshow(transform(img))

    # draw the bounding boxes on the image
    for target in targets:
        box = target['bbox']
        x, y, w, h = box

        label = target['category_id']

        # use different color for different classes
        color = 'r' if label == 1 else 'g'

        rect = plt.Rectangle((x, y), w, h, fill=False, color=color, linewidth=1)
        ax.add_patch(rect)

    plt.show(block=True)

def run(file):
    # get the root directory
    root = os.path.dirname(file)

    # load the dataset
    dataset = CocoDetection(root,
                            file,
                            transforms.ToTensor())

    for img, target in dataset:
        show(img, target)

if __name__ == '__main__':
    """ Example Usage:
    python show.py ./datasets/pklot/images/PUCPR/train/annotations.json
    """
    parser = argparse.ArgumentParser('PKLot Dataset Annotation Viewer')
    parser.add_argument('annotation', type=str, help='The annotation file')
    args = parser.parse_args()

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    run(args.annotation)