import numpy as np
import os
import time
import json
import glob
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.datasets import CocoDetection
from torchvision.transforms import transforms
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.roi_heads import fastrcnn_loss
from torchvision.models.detection.rpn import concat_box_prediction_layers
import matplotlib.pyplot as plt
from torchvision.models.resnet import ResNet50_Weights
from multiprocessing import set_start_method
import PIL
#from PIL import Image
from PIL.ExifTags import TAGS
import argparse

try:
    set_start_method('spawn')
except RuntimeError:
    pass

def isNaN(num):
    return num!= num

# set the device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def load_model(weights_file, num_classes=2):

    # Load the pre-trained ResNet101 model
    model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    num_features = model.fc.in_features

    # Replace the last fully connected layer for fine-tuning
    model.fc = nn.Linear(num_features, num_classes)  # num_classes is the number of output classes

    # use DataParallel to train on multiple GPUs
    model = nn.DataParallel(model)

    # Move the model to the device
    model = model.to(device)

    # set the model to evaluation mode
    model.eval()

    # load the trained model
    model.load_state_dict(torch.load(weights_file, map_location=device))

    # return the model
    return model


def load_boxes(annotation_path):
    """ Load the bounding boxes from the manually annotated dataset """
    # Define the dataset and data loader
    root = os.path.dirname(annotation_path)
    train_dataset = CocoDetection(root,
                            annotation_path,
                            transforms.ToTensor())
    it = iter(train_dataset)
    image, targets = next(it)

    # get all bbox from target
    bboxes = [ item['bbox'] for item in targets ]
    return bboxes, image

def show(bboxes, image):
    """ Plot an image with bounding boxes """
    fig, ax = plt.subplots()
    ax.imshow(transforms.ToPILImage()(image))
    for bbox in bboxes:
        rect = plt.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    plt.show(block=True)

def get_patches(bboxes, image):
    """ Extract patches from an image by giving a list of bounding boxes """
    patches = []
    for bbox in bboxes:
        x, y, w, h = bbox
        patch = image[:, int(y):int(y+h), int(x):int(x+w)]
        patches.append(patch)
    return patches

def show_patches(patches, rows, cols):
    """ Plot a list of patches in a grid of rows x cols """
    # get number of patches to plot
    number_of_patches = cols * rows
    number_of_patches = min(number_of_patches, len(patches))

    # plot each patch
    fig, axs = plt.subplots(rows, cols)

    # Flatten the axes array if it is not already flattened
    if isinstance(axs, np.ndarray):
        axs = axs.flatten()

    # plot each patch
    for i, patch in enumerate(patches):
        # Convert the patch to a PIL image
        image = transforms.ToPILImage()(patch)

        # Convert the image to a NumPy array
        image_np = np.array(image)

        # Convert BGR to RGB
        image_rgb = image_np[..., [0, 1, 2]]

        # Normalize the image
        image_rgb = image_rgb/256

        # Convert the NumPy array back to a PIL image
        # image_rgb = Image.fromarray(image_rgb)

        axs[i].imshow(image_rgb)
        if i == number_of_patches-1:
            break
    plt.show(block=True)

def show_detection(images, predicted_labels, rows, cols):
    # get number of images to plot
    number_of_patches = cols * rows
    number_of_patches = min(number_of_patches, len(images))

    # plot each patch
    fig, axs = plt.subplots(rows, cols)

    classes = ['Space', 'Occupied']

    # Flatten the axes array if it is not already flattened
    if isinstance(axs, np.ndarray):
        axs = axs.flatten()

    # plot each patch
    for i, patch in enumerate(images):
        axs[i].imshow(transforms.ToPILImage()(patch))
        label = classes[predicted_labels[i].item()]
        color = 'r' if label == 'Occupied' else 'g'
        axs[i].text(0, 0, label, bbox=dict(facecolor=color, alpha=0.5))
        plt.axis("off")
        if i == number_of_patches-1:
            break

    # show plot window in maximized mode
    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()
    plt.show(block=True)
    print('done')

# define a dataset for the patches
class PatchesDataset(torch.utils.data.Dataset):
    def __init__(self, patches, transform=None):
        self.patches = patches
        self.transform = transform

    def __getitem__(self, index):
        patch = self.patches[index]
        if self.transform is not None:
            patch = self.transform(patch)

        # resize the patch to 224x224
        patch = transforms.Resize((224, 224))(patch)
        patch = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(patch)
        return patch

    def __len__(self):
        return len(self.patches)

class PatchesDatasetOrigin(torch.utils.data.Dataset):
    def __init__(self, patches, transform=None):
        self.patches = patches
        self.transform = transform

    def __getitem__(self, index):
        patch = self.patches[index]
        if self.transform is not None:
            patch = self.transform(patch)

        # resize the patch to 224x224
        patch = transforms.Resize((224, 224))(patch)
        return patch

    def __len__(self):
        return len(self.patches)


class CocoAnnotationGenerator:
    """ Generate coco annotation file for a list of images and bounding boxes """

    def __init__(self, image_root_path, annotation_path, bboxes, model):
        self.image_root_path = image_root_path
        self.annotation_path = annotation_path
        self.bboxes = bboxes
        self.model = model
        self.annotations = []
        self.root = {}
        self.root['info'] = {
            "description": "PKLot dataset - full annotation",
            "url": "https://app.roboflow.com/personal-ysmui/pklot-full-annotation",
            "version": "1.0",
            "year": 2023,
            "contributor": "University of Buckingham",
            "date_created": "2023-07-04T09:00:00+00:00"
        }
        self.root['licenses'] = [
            {
                "url": "https://creativecommons.org/licenses/by/4.0/",
                "id": 1,
                "name": "Attribution License"
            }
        ]
        self.root['categories'] = [
            {
                "id": 0,
                "name": "spaces",
                "supercategory": "none"
            },
            {
                "id": 1,
                "name": "space-empty",
                "supercategory": "spaces"
            },
            {
                "id": 2,
                "name": "space-occupied",
                "supercategory": "spaces"
            }
        ]
        self.root['images'] = []
        self.root['annotations'] = []

    def add_annotation(self, image_id, bbox, category_id):
        """ Add an annotation to the annotations """
        self.root['annotations'].append({
            "id": len(self.root['annotations']),
            "image_id": image_id,
            "category_id": category_id,
            "segmentation": [],
            "area": bbox[2] * bbox[3],
            "bbox": bbox,
            "iscrowd": 0
        })

    def add_image(self, image, image_path):
        """ Add an image to the annotations """

        id = len(self.root['images'])
        # add the image
        self.root['images'].append({
            "id": id,
            "width": image.width,
            "height": image.height,
            "file_name": image_path,
            "license": 1,
            "flickr_url": "",
            "coco_url": "",
            "date_captured": self.get_timestamp(image, image_path)
        })
        return id

    def get_timestamp(self, image, image_path):
        # Get the EXIF metadata
        exif_data = image._getexif()

        # Check if EXIF data is available
        if exif_data is not None:
            # Iterate over the EXIF tags
            for tag_id, value in exif_data.items():
                tag_name = TAGS.get(tag_id, tag_id)
                if tag_name == "DateTimeOriginal":
                    # The timestamp is stored in the "DateTimeOriginal" tag
                    return value

        # If no EXIF data is available, use the file creation time
        creation_time = os.path.getctime(image.filename)
        time_struct = time.localtime(creation_time)

        # Format the struct_time into a string
        formatted_time = time.strftime("%Y-%m-%dT%H:%M:%S", time_struct)
        return formatted_time

    def dither_boxes(self, boxes, rate=0.1):
        """ Dither the boxes, so that one boxes becomes nine boxes,
            we will generate eight boxes around the original box,
            the result is nin boxes in total.
        """
        new_boxes = []
        for box in boxes:
            x, y, w, h = box
            new_boxes.append([x, y, w, h])
            new_boxes.append([max(0, x - int(rate * w)), max(0, y - int(rate * h)), w, h])
            new_boxes.append([x + int(rate * w), max(0, y - int(rate * h)), w, h])
            new_boxes.append([max(0, x - int(rate * w)), y + int(rate * h), w, h])
            new_boxes.append([x + int(rate * w), y + int(rate * h), w, h])
            new_boxes.append([max(0, x - int(rate * w)), y, w, h])
            new_boxes.append([x + int(rate * w), y, w, h])
            new_boxes.append([x, max(0, y - int(rate * h)), w, h])
            new_boxes.append([x, y + int(rate * h), w, h])
        return new_boxes



    def generate(self):
        """ Generate the annotations """
        # get the list of images
        images = glob.glob(os.path.join(self.image_root_path, "*.jpg"))
        print("found %d images under: %s" % (len(images), self.image_root_path))

        tt = 0

        # for each image
        for image_path in images:
            print("%d: %s" % (tt, image_path))
            # load the image
            image = PIL.Image.open(image_path)

            # add the image to the annotations
            image_id = self.add_image(image, image_path)

            # convert the image to a tensor
            image = transforms.ToTensor()(image)

            # dither the boxes
            dithered_bboxes = self.dither_boxes(self.bboxes)

            # get the patches
            patches = get_patches(dithered_bboxes, image)

            # create a dataset for the patches
            dataset = PatchesDataset(patches)

            # create a dataloader for the dataset
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=36, shuffle=False)

            bbox_index = 0

            img_index = 0

            for imgs in dataloader:

                img_index += 1

                # disable gradient calculation
                with torch.no_grad():
                    output = self.model(imgs)
                    predicted_labels = torch.argmax(output, dim=1)

                dithered_patch_result = []

                # add the annotations
                for i in range(len(predicted_labels)):

                    # get the label
                    label = predicted_labels[i].item()

                    # get the bbox
                    bbox = dithered_bboxes[bbox_index]

                    # add the bbox to the list
                    dithered_patch_result.append((label, bbox))

                    bbox_index += 1

                    if len(dithered_patch_result) == 9:
                        # because we have 9 boxes for each original box

                        # now we will find out all the boxes that are occupied
                        occupied_boxes = [result for result in dithered_patch_result if result[0] == 1]

                        # if there are occupied boxes, we will pick one randomly
                        if len(occupied_boxes) > 0:
                            # pick a random occupied box
                            label, bbox = occupied_boxes[torch.randint(len(occupied_boxes), (1,))]

                        else:
                            # if no box is occupied, we will pick randomly from all boxes
                            label, bbox = dithered_patch_result[torch.randint(len(dithered_patch_result), (1,))]

                        # add the annotation
                        self.add_annotation(image_id, bbox, label+1)
                        dithered_patch_result = []

            tt += 1

        # save the annotations
        with open(self.annotation_path, 'w') as f:
            json.dump(self.root, f, indent=4)


def auto_annotation(args):
    """ Automatically annotate the images in the dataset """

    boxes_annotation = args.boxes_annotation
    image_root = args.image_root
    output_annotation = args.output_annotation
    weights_file = args.weights_file

    # load the image and bboxes
    bboxes, image_template = load_boxes(boxes_annotation)

    # load the model
    model = load_model(weights_file, 2)

    # create a coco annotation generator
    generator = CocoAnnotationGenerator(image_root, output_annotation, bboxes, model)

    # generate the annotations
    generator.generate()


def evaluate_mode(args):

    boxes_annotation = args.boxes_annotation
    weights_file = args.weights_file
    sample_image_path = args.sample_image_path

    # load the image and bboxes
    bboxes, image_template = load_boxes(boxes_annotation)

    # load a test image
    image = PIL.Image.open(sample_image_path)
    image = transforms.ToTensor()(image)

    # get patches from the image
    patches = get_patches(bboxes, image)

    # normalize the patches
    #patches = [ transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(patch) for patch in patches]

    # show the patches
    show_patches(patches, 10, 10)

    # load two datasets, one for the normalized and one for the original images
    dataset = PatchesDataset(patches)
    datasetOrigin = PatchesDatasetOrigin(patches)
    test_data_loader = DataLoader(dataset, batch_size=64, shuffle=False)
    test_data_loader_origin = DataLoader(datasetOrigin, batch_size=64, shuffle=False)

    # define the classes
    classes = ['Background', 'Space', 'Occupied']

    # load the model
    model = load_model(weights_file, 2)

    # run the model on the patches
    for imgs, origin in zip(test_data_loader, test_data_loader_origin):

        # disable gradient calculation
        with torch.no_grad():
            output = model(imgs)
            _, predicted_labels = torch.max(output, dim=1)

        # show the detection
        show_detection(origin, predicted_labels, 8, 8)

    print('done')

if __name__ == '__main__':
    """ Main function
    Usage examples:
        annotate mode:
            python auto-segementation.py annotate
                --weights_file ./resnet50-best.pth
                --boxes_annotation ./datasets/pklot/fully-labeled/PKLot Full Annotation.v3i.coco/test/_annotations.coco.json
                --image_root ./datasets/pklot/images/PUCPR/valid
                --output_annotation ./datasets/pklot/images/PUCPR/valid/wobble_full_annotation.json

        evaluate mode:
            python auto-segementation.py evaluate
                --weights_file ./resnet50-best.pth
                --boxes_annotation ./datasets/pklot/fully-labeled/PKLot Full Annotation.v3i.coco/test/_annotations.coco.json
                --sample_image_path ./datasets/pklot/images/test/2012-09-18_13_40_07_jpg.rf.61c0635e072ebc2d82b7b2ace7b2d673.jpg

    """

    parser = argparse.ArgumentParser('PKLot Dataset Auto Annotation')
    subparsers = parser.add_subparsers(dest="mode", help="Select mode: annotate or show")

    # Annotate mode parser
    annotate_parser = subparsers.add_parser('annotate', help="Annotate mode")
    annotate_parser.add_argument('--weights_file', required=True, help="Path to the weights file")
    annotate_parser.add_argument('--boxes_annotation', required=True, help="Path to bounding boxes annotation file")
    annotate_parser.add_argument('--image_root', required=True, help="Path to the root directory of the images")
    annotate_parser.add_argument('--output_annotation', required=True, help="Path to save the annotations")
    annotate_parser.set_defaults(func=auto_annotation)

    # Evaluate mode parser
    show_parser = subparsers.add_parser('evaluate', help="Evaluate mode")
    show_parser.add_argument('--boxes_annotation', required=True, help="Path to bounding boxes annotation file")
    show_parser.add_argument('--weights_file', required=True, help="Path to the weights file")
    show_parser.add_argument('--sample_image_path', required=True, help="Path to the sample image")
    show_parser.set_defaults(func=evaluate_mode)

    args = parser.parse_args()

    if args.mode is None:
        parser.print_help()
    else:
        args.func(args)
