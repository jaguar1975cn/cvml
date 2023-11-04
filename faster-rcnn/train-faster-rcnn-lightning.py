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

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer

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

def load_model():
    # Load the pre-trained Faster R-CNN model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1)


    # Replace the classifier with a new one that has the correct number of output classes
    num_classes = 3  # Replace with the number of classes in your dataset
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.roi_heads.detections_per_img = 300

    # use DataParallel to train on multiple GPUs
    model = nn.DataParallel(model)
    model.to(device)
    return model

def load_datasets():
    batch_size = 10
    workers = 1

    # Define the dataset and data loader
    train_dataset = CocoDetection('./datasets/pklot/images/train',
                            './datasets/pklot/images/train/full.json',
                            transforms.ToTensor())
    #img_t, _ = train_dataset[0]
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=workers)

    valid_dataset = CocoDetection('./datasets/pklot/images/valid',
                            './datasets/pklot/images/valid/full.json',
                            transforms.ToTensor())
    #img_t, _ = train_dataset[0]
    valid_data_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=workers)
    return (train_data_loader, valid_data_loader)

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
    # Load the pre-trained Faster R-CNN model
    model = load_model()

    # Load the dataset
    train_dataloader, val_dataloader = load_datasets()

    class TrainModule(pl.LightningModule):
         def __init__(self, lr, weight_decay):
             super().__init__()
             self.model = model

             self.lr = lr
             self.weight_decay = weight_decay

         def forward(self, images, targets):
           outputs = self.model(images, targets)
           # return outputs
           # For training, outputs is a dict that contains the losses
           # (for both the RPN and the R-CNN)
           # {
           #   "loss_classifier": ...,  (scalar)
           #   "loss_box_reg": ...,    (scalar)
           #   "loss_objectness": ..., (scalar)
           #   "loss_rpn_box_reg": ..., (scalar)
           # }
           return outputs

         def common_step(self, batch, batch_idx):
           pixel_values = batch["pixel_values"]
           pixel_mask = batch["pixel_mask"]
           labels = [{k: v.to(self.device) for k, v in t.items()} for t in batch["labels"]]

           outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)

           loss = outputs.loss
           loss_dict = outputs.loss_dict

           return loss, loss_dict

         def training_step(self, batch, batch_idx):

            images, targets = batch

            loss_dict = self.model(images, targets)
            loss = sum(loss for loss in loss_dict.values())

            # logs metrics for each training_step,
            # and the average across the epoch
            self.log("training_loss", loss)
            for k,v in loss_dict.items():
              self.log("train_" + k, v.item())

            return loss

         def validation_step(self, batch, batch_idx):
            images, targets = batch

            loss_dict, detections  = eval_forward(self.model, images, targets)
            loss = sum(loss for loss in loss_dict.values())

            # logs metrics for each training_step,
            # and the average across the epoch
            self.log("validation_loss", loss)
            for k,v in loss_dict.items():
              self.log("validation_" + k, v.item())

            return loss

         def configure_optimizers(self):
            # Define the optimizer and learning rate scheduler
            params = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
            return optimizer

         def train_dataloader(self):
            return train_dataloader

         def val_dataloader(self):
            return val_dataloader

    cats = train_dataloader.dataset.coco.cats
    id2label = {k: v['name'] for k,v in cats.items()}
    label2id = {(v['name']): k for k,v in cats.items()}
    print(id2label)
    print(label2id)

    model = TrainModule(lr=1e-4, weight_decay=1e-4)

    root_folder = "faster-rcnn_logs"

    # Define a logger with a custom directory
    logger = TensorBoardLogger(save_dir=root_folder, name='logs')

    trainer = Trainer(max_epochs=30, gradient_clip_val=0.1, logger=logger, default_root_dir=root_folder)
    trainer.fit(model)

    model.model.push_to_hub("jameszeng/faster-rcnn-finetuned-pklot-full", private=True)


if __name__ == '__main__':
    train()
