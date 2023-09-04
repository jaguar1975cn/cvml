import numpy as np
import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.datasets import CocoDetection
from torchvision.transforms import transforms
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.roi_heads import fastrcnn_loss
from torchvision.models.detection.rpn import concat_box_prediction_layers

import datetime
import os
from typing import Tuple, List, Dict, Optional
from collections import OrderedDict

from multiprocessing import set_start_method

try:
    set_start_method('spawn')
except RuntimeError:
    pass


# only use GPU 1,2
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
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


def save_ckp(state, is_best, checkpoint_path, best_model_path):
    """
    state: checkpoint we want to save
    is_best: is this the best checkpoint; min validation loss
    checkpoint_path: path to save checkpoint
    best_model_path: path to save best model
    """
    f_path = checkpoint_path
    # save checkpoint data to the path given, checkpoint_path
    torch.save(state, f_path)
    # if it is a best model, min validation loss
    if is_best:
        # only save the model state dict
        torch.save(state['state_dict'], best_model_path)

def load_ckp(checkpoint_fpath, model, scheduler):
    """
    checkpoint_path: path to save checkpoint
    model: model that we want to load checkpoint parameters into
    scheduler: scheduler we defined in previous training
    """
    # load check point
    checkpoint = torch.load(checkpoint_fpath)
    # initialize state_dict from checkpoint to model
    model.load_state_dict(checkpoint['state_dict'])
    # initialize optimizer from checkpoint to optimizer
    scheduler.load_state_dict(checkpoint['scheduler'])
    # initialize valid_loss_min from checkpoint to valid_loss_min
    valid_loss_min = checkpoint['valid_loss_min']
    # return model, optimizer, epoch value, min validation loss
    return model, scheduler, checkpoint['epoch'], valid_loss_min.item()

def eval_forward(model, images, targets):
    # type: (nn.Module, List[torch.Tensor], Optional[List[Dict[str, torch.Tensor]]]) -> Tuple[Dict[str, torch.Tensor], List[Dict[str, torch.Tensor]]]
    """
    Args:
        model (nn.Module): model to evaluate
        images (list[Tensor]): images to be processed
        targets (list[Dict[str, Tensor]]): ground-truth boxes present in the image (optional)
    Returns:
        result (list[BoxList] or dict[Tensor]): the output from the model.
            It returns list[BoxList] contains additional fields
            like `scores`, `labels` and `mask` (for Mask R-CNN models).
    """
    model.eval()

    # our model is wrapped in DataParallel, get the underlying module (faster-rcnn)
    model = model.module

    original_image_sizes: List[Tuple[int, int]] = []
    for img in images:
        val = img.shape[-2:]
        assert len(val) == 2
        original_image_sizes.append((val[0], val[1]))

    images, targets = model.transform(images, targets)

    # Check for degenerate boxes
    # TODO: Move this to a function
    if targets is not None:
        for target_idx, target in enumerate(targets):
            boxes = target["boxes"]
            degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
            if degenerate_boxes.any():
                # print the first degenerate box
                bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                degen_bb: List[float] = boxes[bb_idx].tolist()
                raise ValueError(
                    "All bounding boxes should have positive height and width."
                    f" Found invalid box {degen_bb} for target at index {target_idx}."
                )

    features = model.backbone(images.tensors)
    if isinstance(features, torch.Tensor):
        features = OrderedDict([("0", features)])
    model.rpn.training=True
    #model.roi_heads.training=True

    #####proposals, proposal_losses = model.rpn(images, features, targets)
    features_rpn = list(features.values())
    objectness, pred_bbox_deltas = model.rpn.head(features_rpn)
    anchors = model.rpn.anchor_generator(images, features_rpn)

    num_images = len(anchors)
    num_anchors_per_level_shape_tensors = [o[0].shape for o in objectness]
    num_anchors_per_level = [s[0] * s[1] * s[2] for s in num_anchors_per_level_shape_tensors]
    objectness, pred_bbox_deltas = concat_box_prediction_layers(objectness, pred_bbox_deltas)
    # apply pred_bbox_deltas to anchors to obtain the decoded proposals
    # note that we detach the deltas because Faster R-CNN do not backprop through
    # the proposals
    proposals = model.rpn.box_coder.decode(pred_bbox_deltas.detach(), anchors)
    proposals = proposals.view(num_images, -1, 4)
    proposals, scores = model.rpn.filter_proposals(proposals, objectness, images.image_sizes, num_anchors_per_level)

    proposal_losses = {}
    assert targets is not None
    labels, matched_gt_boxes = model.rpn.assign_targets_to_anchors(anchors, targets)
    regression_targets = model.rpn.box_coder.encode(matched_gt_boxes, anchors)
    loss_objectness, loss_rpn_box_reg = model.rpn.compute_loss(
        objectness, pred_bbox_deltas, labels, regression_targets
    )
    proposal_losses = {
        "loss_objectness": loss_objectness,
        "loss_rpn_box_reg": loss_rpn_box_reg,
    }

    #####detections, detector_losses = model.roi_heads(features, proposals, images.image_sizes, targets)
    image_shapes = images.image_sizes
    proposals, matched_idxs, labels, regression_targets = model.roi_heads.select_training_samples(proposals, targets)
    box_features = model.roi_heads.box_roi_pool(features, proposals, image_shapes)
    box_features = model.roi_heads.box_head(box_features)
    class_logits, box_regression = model.roi_heads.box_predictor(box_features)

    result: List[Dict[str, torch.Tensor]] = []
    detector_losses = {}
    loss_classifier, loss_box_reg = fastrcnn_loss(class_logits, box_regression, labels, regression_targets)
    detector_losses = {"loss_classifier": loss_classifier, "loss_box_reg": loss_box_reg}
    boxes, scores, labels = model.roi_heads.postprocess_detections(class_logits, box_regression, proposals, image_shapes)
    num_images = len(boxes)
    for i in range(num_images):
        result.append(
            {
                "boxes": boxes[i],
                "labels": labels[i],
                "scores": scores[i],
            }
        )
    detections = result
    detections = model.transform.postprocess(detections, images.image_sizes, original_image_sizes)  # type: ignore[operator]
    model.rpn.training=False
    model.roi_heads.training=False
    losses = {}
    losses.update(detector_losses)
    losses.update(proposal_losses)
    return losses, detections

def train():
    checkpoint_path = 'faster-rcnn-checkpoint.pt'
    best_model_path = 'faster-rcnn-best.pt'
    batch_size = 10
    workers = 1

    # Define the dataset and data loader
    train_dataset = CocoDetection('./datasets/pklot/images/train',
                            './datasets/pklot/images/PUCPR/train/annotations.json',
                            transforms.ToTensor())
    #img_t, _ = train_dataset[0]
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=workers)

    valid_dataset = CocoDetection('./datasets/pklot/images/valid',
                            './datasets/pklot/images/PUCPR/valid/annotations.json',
                            transforms.ToTensor())
    #img_t, _ = train_dataset[0]
    valid_data_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=workers)

    # Load the pre-trained Faster R-CNN model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1)


    # Replace the classifier with a new one that has the correct number of output classes
    num_classes = 2  # Replace with the number of classes in your dataset
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.roi_heads.detections_per_img = 300

    # use DataParallel to train on multiple GPUs
    model = nn.DataParallel(model)
    model.to(device)

    # Define the optimizer and learning rate scheduler
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    valid_losses_min = np.Inf

    # Train the model
    num_epochs = 30  # Replace with the number of epochs you want to train for
    for epoch in range(num_epochs):

        # Set the model to training mode
        model.train()

        b=0
        for images, targets in train_data_loader:
            b=b+1
            optimizer.zero_grad()
            # print(targets)
            loss_dict = model(images, targets)
            train_losses = sum(loss for loss in loss_dict.values())
            # when using DataParallel, the loss is partial loss computed on each gpu, so we need to use losses.sum()
            train_losses.sum().backward()
            optimizer.step()
            print('Train: {} Epoch {}, batch {}, Training loss {}'.format(datetime.datetime.now(), epoch, b, train_losses / len(images)))

        # update the learning rate
        lr_scheduler.step()

        # Validation
        epoch_valid_losses = 0
        b=0
        model.eval()
        for images, targets in valid_data_loader:
            b=b+1
            with torch.no_grad():
                valid_loss_dict, detections = eval_forward(model, images, targets)

            valid_losses = sum(loss for loss in valid_loss_dict.values())
            epoch_valid_losses = epoch_valid_losses + valid_losses
            # when using DataParallel, the loss is partial loss computed on each gpu, so we need to use losses.sum()
            print('Valid: {} Epoch {}, batch {}, Validation loss {}'.format(datetime.datetime.now(), epoch, b, valid_losses / len(images)))

        # save checkpoint
        checkpoint = {
            'epoch': epoch + 1,
            'valid_loss_min': valid_losses_min,
            'state_dict': model.state_dict(),
            'scheduler': lr_scheduler.state_dict(),
        }

        # save checkpoint
        save_ckp(checkpoint, False, checkpoint_path, best_model_path)

        if epoch_valid_losses <= valid_losses_min:
            print('Validation losses decreased ({:.6f} --> {:.6f}).  Saving model to {}'.format(valid_losses_min, epoch_valid_losses, best_model_path))
            # save checkpoint as best model
            save_ckp(checkpoint, True, checkpoint_path, best_model_path)
            valid_losses_min = epoch_valid_losses

    # Save the trained model
    torch.save(model.state_dict(), 'faster-rcnn-model.pt')


if __name__ == '__main__':
    train()
