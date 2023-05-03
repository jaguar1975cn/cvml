import torch
import torchvision
import torch.nn as nn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
import matplotlib.pyplot as plt
import torchvision.transforms as T
from PIL import Image
import os
import glob

# set the device
os.environ['CUDA_VISIBLE_DEVICES'] = '1,2'
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def load_model():

    # load the pre-trained model
    #model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1)
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)

    # Replace the classifier with a new one that has the correct number of output classes
    num_classes = 3  # Replace with the number of classes in your dataset
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # use DataParalle on model
    model = nn.DataParallel(model)

    # move it to device
    model.to(device)

    # set the model to evaluation mode
    model.eval()

    # load the trained model
    model.load_state_dict(torch.load('best.pt', map_location=device))
    return model


def detect(model, img_path):

    # load the image
    img = Image.open(img_path)

    # convert to tensor
    convert_tensor = T.ToTensor()
    tensor=convert_tensor(img)

    # make it 3 channels and 1 batch
    batch = tensor.unsqueeze(0)


    # Run the image through the model
    # disable gradient calculation
    with torch.no_grad():
        output = model(batch)

    # Print the predicted classes and bounding boxes
    output = output[0]

    # find count of each class
    count = torch.bincount(output['labels'])
    # print(count)

    if len(count)<=1:
        print("No object detected")
        return

    if len(count)==2:
        print("Found {} boxes and 0 spaces".format(count[1]))
    else:
        print("Found {} boxes and {} spaces".format(count[1], count[2]))

    # define the classes
    classes = ['background', 'unoccupied', 'occupied']

    # Plot the image with the bounding boxes

    fig, ax = plt.subplots()

    ax.imshow(img)

    index = 0

    # draw the bounding boxes on the image
    for box in output['boxes']:
        xmin, ymin, xmax, ymax = box.cpu()  

        label = classes[output['labels'][index]]

        # use different color for different classes
        color = 'r' if label == 'occupied' else 'g'

        rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, color=color, linewidth=1)
        ax.add_patch(rect)
        index = index + 1


    plt.savefig('output/' + os.path.basename(img_path))
    plt.close()


if __name__ == '__main__':
    # set the dataset root
    root = 'datasets/pklot/images/test/'
    model = load_model()
    for filename in glob.glob(root+'/*.jpg'):
        detect(model, filename)
