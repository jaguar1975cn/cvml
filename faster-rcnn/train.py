import torch
import torchvision
import torch.nn as nn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.datasets import CocoDetection
from torchvision.transforms import transforms
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from torch.utils.data import DataLoader
import datetime
import os

from multiprocessing import set_start_method
try:
    set_start_method('spawn')
except RuntimeError:
    pass


# only use GPU 1,2
os.environ['CUDA_VISIBLE_DEVICES'] = '1,2'
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


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


def train():
    # Define the dataset and data loader
    dataset = CocoDetection('./datasets/pklot/images/train',
                            './datasets/pklot/images/train/_annotations.coco.json',
                            transforms.ToTensor())
    img_t, _ = dataset[0]
    data_loader = DataLoader(dataset, batch_size=10, shuffle=False, collate_fn=collate_fn, num_workers=4)

    # Load the pre-trained Faster R-CNN model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1)


    # Replace the classifier with a new one that has the correct number of output classes
    num_classes = 3  # Replace with the number of classes in your dataset
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # use DataParallel to train on multiple GPUs
    model = nn.DataParallel(model)
    model.to(device)

    # Set the model to training mode
    model.train()

    # Define the optimizer and learning rate scheduler
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # Train the model
    num_epochs = 30  # Replace with the number of epochs you want to train for
    for epoch in range(num_epochs):
        b=0
        for images, targets in data_loader:
            b=b+1
            optimizer.zero_grad()
            # print(targets)
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            # when using DataParallel, the loss is partial loss computed on each gpu, so we need to use losses.sum()
            losses.sum().backward()
            optimizer.step()
            print('{} Epoch {}, batch {}, Training loss {}'.format(datetime.datetime.now(), epoch, b, losses / len(data_loader)))
        lr_scheduler.step()

    # Save the trained model
    torch.save(model.state_dict(), 'model.pt')


if __name__ == '__main__':
    train()
