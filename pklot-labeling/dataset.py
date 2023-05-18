import os
import pandas as pd
import torch
from torch.utils.data import Dataset
import PIL
from torchvision.io import read_image

class PklotSegmentedDataset(Dataset):
    """ PKLot dataset with segmentation masks.

    classes = ['Empty', 'Occupied']

    Dataset file structure:
    - img_dir
        - Cloudy
            - 2012-09-12
                - Empty
                - Occupied
        - Rainy
            - 2012-09-12
                - Empty
                - Occupied
        - Sunny
            - 2012-09-12
                - Empty
                - Occupied
    """

    def __init__(self, img_dir, transform=None, target_transform=None):
        """
        Args:
            img_dir (string): Directory with all the images.
        """
        self.img_dir = img_dir
        self.parse_dir()
        self.transform = transform
        self.target_transform = target_transform

    def parse_dir(self):
        """ Enumerate all subdirectories and parse them.

        item = {
            "img_name": "2012-09-12_06_05_16__001.jpg",
            "img_label": "Empty",
            "img_path": "./Cloudy/2012-09-12/Empty/2012-09-12_06_05_16__001.jpg",
            "img_weather": "Cloudy",
            }
        """
        self.images = []

        for weather in os.listdir(self.img_dir):
            weather_dir = os.path.join(self.img_dir, weather)
            if os.path.isdir(weather_dir):
                for date in os.listdir(weather_dir):
                    date_dir = os.path.join(weather_dir, date)
                    if os.path.isdir(date_dir):
                        for label in os.listdir(date_dir):
                            label_dir = os.path.join(date_dir, label)
                            if os.path.isdir(label_dir):
                                for img_name in os.listdir(label_dir):
                                    img_path = os.path.join(label_dir, img_name)
                                    item = {
                                        "img_name": img_name,
                                        "img_label": label,
                                        "img_path": img_path,
                                        "img_weather": weather,
                                    }
                                    self.images.append(item)

    def __len__(self):
        """ Return the length of the dataset. """
        return len(self.images)

    def __getitem__(self, idx):
        """ Return the item at index idx. """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # get the image path and label
        img_path = self.images[idx]["img_path"]
        img_label = self.images[idx]["img_label"]
        img_weather = self.images[idx]["img_weather"]

        # read the image
        #image = read_image(img_path)
        image = PIL.Image.open(img_path)

        # apply the image transform
        if self.transform:
            image = self.transform(image)

        # create the target
        target = {
            "class": img_label,
            "label": 0 if img_label == "Empty" else 1,
            "weather": img_weather,
        }

        # apply the target transform
        if self.target_transform:
            target = self.target_transform(target)

        return image, target