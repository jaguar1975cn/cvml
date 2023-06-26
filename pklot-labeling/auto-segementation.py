import numpy as np
import torch
import torchvision
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
import matplotlib.patches as patches
from torchvision.models.resnet import ResNet50_Weights
import PIL

def load_model(num_classes=2):

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    #model.load_state_dict(torch.load('resnet50.pth', map_location=device))
    model.load_state_dict(torch.load('resnet50-best.pth', map_location=device))

    # return the model
    return model


def load():
    # Define the dataset and data loader
    train_dataset = CocoDetection('/home/zengyi/ssd/MSc/cvml/datasets/pklot/fully-labeled/PKLot Full Annotation.v2i.coco/test',
                            '/home/zengyi/ssd/MSc/cvml/datasets/pklot/fully-labeled/PKLot Full Annotation.v2i.coco/test/_annotations.coco.json',
                            transforms.ToTensor())
    it = iter(train_dataset)
    image, targets = next(it)

    # get all bbox from target
    bboxes = [ item['bbox'] for item in targets ]
    return bboxes, image

def show(bboxes, image):
    # plot each bbox of the image
    fig, ax = plt.subplots()
    ax.imshow(transforms.ToPILImage()(image))
    for bbox in bboxes:
        rect = plt.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    plt.show()

def get_patches(bboxes, image):
    # extract each patch from the image
    patches = []
    for bbox in bboxes:
        x, y, w, h = bbox
        patch = image[:, int(y):int(y+h), int(x):int(x+w)]
        patches.append(patch)
    return patches

def show_patches(patches, rows, cols):
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
        axs[i].imshow(transforms.ToPILImage()(patch))
        if i == number_of_patches-1:
            break
    plt.show()

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
    plt.show()
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

if __name__ == '__main__':

    # load the image and bboxes
    bboxes, image = load()

    # get patches from the image
    #patches = get_patches(bboxes, image)

    # load another image
    #image = PIL.Image.open('/home/zengyi/ssd/MSc/cvml/datasets/pklot/images/train/2012-09-12_07_49_42_jpg.rf.e7098b35dc482d8fb1535974280d1df2.jpg')
    # image = PIL.Image.open('/home/zengyi/ssd/MSc/cvml/datasets/pklot/images/train/2012-09-20_17_19_42_jpg.rf.44722699775b1a476e1506643b874401.jpg')
    image = PIL.Image.open('/home/zengyi/ssd/MSc/cvml/datasets/pklot/images/test/2012-09-18_13_40_07_jpg.rf.61c0635e072ebc2d82b7b2ace7b2d673.jpg')
    image = transforms.ToTensor()(image)

    patches = get_patches(bboxes, image)

    dataset = PatchesDataset(patches)
    datasetOrigin = PatchesDatasetOrigin(patches)
    test_data_loader = DataLoader(dataset, batch_size=64, shuffle=False)
    test_data_loader_origin = DataLoader(datasetOrigin, batch_size=64, shuffle=False)

    # show(bboxes, image)
    #show_patches(patches, 10, 10)

    classes = ['Background', 'Space', 'Occupied']

    model = load_model(2)

    for imgs, origin in zip(test_data_loader, test_data_loader_origin):
        # Run the image through the model
        # disable gradient calculation
        with torch.no_grad():
            output = model(imgs)
            _, predicted_labels = torch.max(output, dim=1)

        # show the detection
        show_detection(origin, predicted_labels, 8, 8)

    print('done')