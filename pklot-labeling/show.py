import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.datasets import CocoDetection
from torchvision.transforms import transforms
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.roi_heads import fastrcnn_loss
from torchvision.models.detection.rpn import concat_box_prediction_layers
import matplotlib.pyplot as plt
import torchvision.transforms as T

def show(img, targets):
    # define the classes
    classes = ['background', 'unoccupied', 'occupied']

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

    fig.show()

    plt.close(fig)

def run():
    train_dataset = CocoDetection('./',
                            './datasets/pklot/images/PUCPR/train/annotations.json',
                            transforms.ToTensor())

    #train_data_loader = DataLoader(train_dataset, batch_size=10, shuffle=False, num_workers=4)
    for img, target in train_dataset:
        show(img, target)

if __name__ == '__main__':
    run()