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
import glob
from torchvision.datasets import CocoDetection

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
        color = 'r' if label == 'occupied' else 'g'

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

        # use different color for different classes
        color = 'r' if label == 'occupied' else 'g'

        rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, color=color, linewidth=1)
        ax.add_patch(rect)

        index = index + 1

    fig.show()
    #plt.savefig('output/{}.jpg'.format((index)))
    plt.close(fig)

def detect(index, model, img:torch.Tensor, target):

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
    # print(count)

    if len(count)<=1:
        print("No object detected")
        return

    if len(count)==2:
        print("{}) Found {} spaces and 0 cars".format(index, count[1]))
    else:
        print("{}) Found {} spaces and {} cars".format(index, count[1], count[2]))

    # show(img, output, target)

    classes = [1,2]
    # collect boxes per class
    predictions = []
    for c in classes:
        boxes = output['boxes'][output['labels'] == c]
        scores = output['scores'][output['labels'] == c]
        x = []
        for b,s in zip(boxes, scores):
            x.append((b.cpu().tolist(), s.cpu().item()))

        predictions.append(x)

    ground_truths = []
    for c in classes:
        boxes = target['boxes'][target['labels'] == c]
        ground_truths.append(boxes)


    ap = mean_average_precision(predictions, ground_truths, iou_threshold=0.5)
    print("{}) Mean AP: {:.4f}".format(index, ap))

    # compute the APs for each class
    return ap


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
        else:
            target_dict["boxes"] = torch.tensor([ [t['bbox'][0], t['bbox'][1], t['bbox'][0] + t['bbox'][2], t['bbox'][1] + t['bbox'][3] ] for t in target], dtype=torch.float32).to(device)
            target_dict["labels"] = torch.tensor([t['category_id'] for t in target]).to(device)

        targets.append(target_dict)

    images = torch.stack(images, dim=0)

    return images, targets

def iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = box1_area + box2_area - intersection_area
    return intersection_area / union_area


def average_precision(predictions, ground_truths, iou_threshold):
    if len(predictions) == 0 and len(ground_truths) == 0:
        return 1
    predictions = sorted(predictions, key=lambda x: x[1], reverse=True)

    true_positives = torch.zeros(len(predictions))
    false_positives = torch.zeros(len(predictions))
    num_ground_truths = len(ground_truths)

    for i, pred in enumerate(predictions):
        best_iou = 0
        best_gt_idx = -1
        for j, gt in enumerate(ground_truths):
            current_iou = iou(pred[0], gt)
            if current_iou > best_iou:
                best_iou = current_iou
                best_gt_idx = j

        if best_iou >= iou_threshold:
            true_positives[i] = 1
            ground_truths.pop(best_gt_idx)
        else:
            false_positives[i] = 1

    cumulative_true_positives = torch.cumsum(true_positives, dim=0)
    cumulative_false_positives = torch.cumsum(false_positives, dim=0)
    precision = cumulative_true_positives / (cumulative_true_positives + cumulative_false_positives)
    recall = cumulative_true_positives / num_ground_truths

    precision = torch.cat((torch.tensor([1]), precision))
    recall = torch.cat((torch.tensor([0]), recall))

    ap = torch.sum((recall[1:] - recall[:-1]) * precision[1:])
    return ap.item()


def mean_average_precision(predictions_per_class, ground_truths_per_class, iou_threshold=0.5):
    aps = []
    for pred, gt in zip(predictions_per_class, ground_truths_per_class):
        aps.append(average_precision(pred, gt.tolist(), iou_threshold))

    return sum(aps) / len(aps)

if __name__ == '__main__':
    # set the dataset root
    root = 'datasets/pklot/images/test'
    test_dataset = CocoDetection(root,  root+'/_annotations.coco.json', T.ToTensor())

    test_data_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn, num_workers=1)

    model = load_model()

    aps = []

    index = 0

    for imgs, target in test_data_loader:
        img = imgs[0]
        aps.append(detect(index, model, img, target[0]))
        index += 1

    mAP = sum(aps) / len(aps)
    print("Final Mean AP: {:.4f}".format(Map))

