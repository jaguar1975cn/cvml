import torch
import torchvision
import torch.nn as nn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
import matplotlib.pyplot as plt
import torchvision.transforms as T
from PIL import Image
import os

# set the device
os.environ['CUDA_VISIBLE_DEVICES'] = '1,2'
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# set the dataset root
root = 'datasets/pklot/images/test/'

# load the image
img = Image.open(root + '2013-02-22_11_35_06_jpg.rf.6729808dc9b4633b2e68455ced4cac4a.jpg')

# convert to tensor
convert_tensor = T.ToTensor()
tensor=convert_tensor(img)

# make it 3 channels and 1 batch
batch = tensor.unsqueeze(0)

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
model.load_state_dict(torch.load('faster-rcnn/model.pt', map_location=device))

# Run the image through the model
# disable gradient calculation
with torch.no_grad():
    output = model(batch)

# Print the predicted classes and bounding boxes
output = output[0]
print(output['labels'])
print(output['boxes'])

# define the classes
classes = ['background', 'occupied', 'unoccupied']

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


plt.savefig('output/output.png')
#plt.show()
