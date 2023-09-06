import torch
import torchvision
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from torchvision.ops import box_iou
import matplotlib.pyplot as plt
import torchvision.transforms as T
from PIL import Image
import os
from torchvision.datasets import CocoDetection
from multiprocessing import set_start_method
from pycocotools.cocoeval import COCOeval
from coco_eval import CocoEvaluator

try:
    set_start_method('spawn')
except RuntimeError:
    pass

def isNaN(num):
    return num!= num

# set the device
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def load_model():

    # load the pre-trained model
    #model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1)
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)

    # Replace the classifier with a new one that has the correct number of output classes
    num_classes = 2  # Replace with the number of classes in your dataset
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.roi_heads.detections_per_img = 300

    # use DataParalle on model
    model = nn.DataParallel(model)

    # move it to device
    model.to(device)

    # set the model to evaluation mode
    model.eval()

    # load the trained model
    model.load_state_dict(torch.load('faster-rcnn/faster-rcnn-best.pt', map_location=device))
    return model

def show(img, output, target):
    # define the classes
    classes = ['background', 'unoccupied', 'occupied']

    # Plot the image with the bounding boxes

    fig, axs = plt.subplots(1, 2)

    transform = T.ToPILImage()
    ax = axs[0]
    ax.imshow(transform(img))

    index = 0

    # draw the bounding boxes on the image
    for box in output['boxes']:
        xmin, ymin, xmax, ymax = box.cpu()

        label = classes[output['labels'][index]]

        # use different color for different classes
        if output['labels'][index] == 0:
            color = 'b'
        elif output['labels'][index] == 1:
            color = 'g'
        else:
            color = 'r'
        #color = 'r' if label == 'occupied' else 'g'

        rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, color=color, linewidth=1)
        ax.add_patch(rect)
        # score = '{:.4f}'.format(output['scores'][index].item())
        # ax.text(xmin, ymin, score, bbox=dict(facecolor=color, alpha=0.5))

        index = index + 1

    ax = axs[1]
    ax.imshow(transform(img))

    index = 0

    # draw the bounding boxes on the image
    for box in target['boxes']:
        xmin, ymin, xmax, ymax = box

        label = classes[target['labels'][index]]

        if target['labels'][index] == 0:
            color = 'b'
        elif target['labels'][index] == 1:
            color = 'g'
        else:
            color = 'r'

        # use different color for different classes
        #color = 'r' if label == 'occupied' else 'g'

        rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, color=color, linewidth=1)
        ax.add_patch(rect)

        index = index + 1

    fig.show()
    #plt.savefig('output2/{}.jpg'.format(target["image_id"]))
    plt.close(fig)

def detect(index, model, img:torch.Tensor, target):

    # the iou thresholds to consider
    thresholds = torch.arange(start=0.2, end=0.7, step=0.05)

    # make it 3 channels and 1 batch
    batch = img.unsqueeze(0)

    # Run the image through the model
    # disable gradient calculation
    with torch.no_grad():
        output = model(batch)

    # Print the predicted classes and bounding boxes
    output = output[0]

    # find count of each class
    count = torch.bincount(output['labels'])

    if len(count)<=1:
        print("No object detected")
        if len(target['boxes'])==0:
            # no ground truth, the AP is undefined
            return {}

        # no object detected, but there are ground truth, the AP is 0
        return {}

    if len(count)==2:
        print("{}) Image #{}: found {} spaces and {} cars".format(index, target["image_id"], count[0], count[1]))

    res = {target["image_id"]: output}

    return res

def collate_fn(batch):
    images, targets = [], []

    for item in batch:
        image, target = item
        images.append(image.to(device))

        # Convert the target to the format expected by the model
        # fasterrcnn_resnet50_fpn:
        # The model expects both the input tensors and a targets (list of dictionary), containing:
        # boxes (FloatTensor[N, 4]): the ground-truth boxes in [x1, y1, x2, y2] format, with 0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H.
        # labels (Int64Tensor[N]): the class label for each ground-truth box
        #
        # In coco annotation, the bbox is: [x,y,width,height]
        target_dict = {}
        if not target:
            target_dict["boxes"] = torch.empty((0, 4), dtype=torch.float32).to(device)
            target_dict["labels"] = torch.empty((0), dtype=torch.int64).to(device)
            target_dict["image_id"] = -1
        else:
            target_dict["boxes"] = torch.tensor([ [t['bbox'][0], t['bbox'][1], t['bbox'][0] + t['bbox'][2], t['bbox'][1] + t['bbox'][3] ] for t in target], dtype=torch.float32).to(device)
            target_dict["labels"] = torch.tensor([t['category_id'] for t in target]).to(device)
            target_dict["image_id"] = target[0]["image_id"]

        targets.append(target_dict)

    images = torch.stack(images, dim=0)

    return images, targets

if __name__ == '__main__':
    # set the dataset root
    root = 'datasets/pklot/images/PUCPR/test'
    #test_dataset = CocoDetection(root,  root+'/annotations.json', T.ToTensor())
    test_dataset = CocoDetection(root,  root+'/has-cars.json', T.ToTensor())

    test_data_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn, num_workers=1)

    model = load_model()

    results = []

    index = 0

    coco_evaluator = CocoEvaluator(test_dataset.coco, iou_types=["bbox"], maxDets=300)

    for imgs, target in test_data_loader:
        img = imgs[0]
        batch_dt = detect(index, model, img, target[0])

        if batch_dt:
            coco_evaluator.update(batch_dt)

        index += 1
        # if index > 3:
        #     break

    coco_evaluator.synchronize_between_processes()
    coco_evaluator.accumulate()
    coco_evaluator.summarize()

    # the coco_dt should be in the following format:
    # coco_dt = [
    # {
    #     'image_id': 1,
    #     'category_id': 3,
    #     'bbox': [100, 150, 50, 60],
    #     'score': 0.95
    # },
    # {
    #     'image_id': 1,
    #     'category_id': 2,
    #     'bbox': [200, 180, 40, 50],
    #     'score': 0.90
    # }
    #]
    #
    # # save coco_dt into json file
    # with open('results.json', 'w') as f:
    #     json.dump(coco_dt, f)

    # coco_dt = test_dataset.coco.loadRes('results.json')

    # coco_eval = COCOeval(test_dataset.coco, coco_dt, iouType="bbox") # initialize CocoEval object
    # coco_eval.params.maxDets = [1, 10, 300]
    # # Run evaluation
    # coco_eval.evaluate()
    # coco_eval.accumulate()
    # coco_eval.summarize()
