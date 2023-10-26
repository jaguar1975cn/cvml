from multiprocessing import set_start_method

import torchvision
import os
import numpy as np
import os
from PIL import Image, ImageDraw
import pytorch_lightning as pl
from transformers import DeformableDetrConfig, DeformableDetrForObjectDetection
import torch
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader

class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, processor):
        ann_file = os.path.join(img_folder, "full.json")
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self.processor = processor

    def __getitem__(self, idx):
        # read in PIL image and target in COCO format
        # feel free to add data augmentation here before passing them to the next step
        img, target = super(CocoDetection, self).__getitem__(idx)

        # preprocess image and target (converting target to DETR format, resizing + normalization of both image and target)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        encoding = self.processor(images=img, annotations=target, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze() # remove batch dimension
        target = encoding["labels"][0] # remove batch dimension

        return pixel_values, target



def main():
    from transformers import DeformableDetrImageProcessor
    from pytorch_lightning.loggers import TensorBoardLogger

    pretrained = "SenseTime/deformable-detr"

    processor = DeformableDetrImageProcessor.from_pretrained(pretrained)

    train_dataset = CocoDetection(img_folder='../datasets/pklot/images/train', processor=processor)
    val_dataset = CocoDetection(img_folder='../datasets/pklot/images/valid', processor=processor)

    print("Number of training examples:", len(train_dataset))
    print("Number of validation examples:", len(val_dataset))

    image_ids = train_dataset.coco.getImgIds()

    cats = train_dataset.coco.cats
    id2label = {k: v['name'] for k,v in cats.items()}
    label2id = {(v['name']): k for k,v in cats.items()}
    print(id2label)
    print(label2id)

    def collate_fn(batch):
        pixel_values = [item[0] for item in batch]
        encoding = processor.pad(pixel_values, return_tensors="pt")
        labels = [item[1] for item in batch]
        batch = {}
        batch['pixel_values'] = encoding['pixel_values']
        batch['pixel_mask'] = encoding['pixel_mask']
        batch['labels'] = labels
        return batch

    train_dataloader = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=2, shuffle=True)
    val_dataloader = DataLoader(val_dataset, collate_fn=collate_fn, batch_size=2)
    batch = next(iter(train_dataloader))

    print(batch.keys())
    pixel_values, target = train_dataset[0]
    print(pixel_values.shape)

    print(target)
    class Detr(pl.LightningModule):
         def __init__(self, lr, lr_backbone, weight_decay, id2label, label2id):
             super().__init__()
             # replace COCO classification head with custom head
             # we specify the "no_timm" variant here to not rely on the timm library
             # for the convolutional backbone
             self.model = DeformableDetrForObjectDetection.from_pretrained(pretrained,
                                                                 num_labels=len(id2label),
                                                                 id2label=id2label,
                                                                 label2id=label2id,
                                                                 ignore_mismatched_sizes=True)
             # see https://github.com/PyTorchLightning/pytorch-lightning/pull/1896
             self.lr = lr
             self.lr_backbone = lr_backbone
             self.weight_decay = weight_decay

         def forward(self, pixel_values, pixel_mask):
           outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)

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
            loss, loss_dict = self.common_step(batch, batch_idx)     
            # logs metrics for each training_step,
            # and the average across the epoch
            self.log("training_loss", loss)
            for k,v in loss_dict.items():
              self.log("train_" + k, v.item())

            return loss

         def validation_step(self, batch, batch_idx):
            loss, loss_dict = self.common_step(batch, batch_idx)     
            self.log("validation_loss", loss)
            for k,v in loss_dict.items():
              self.log("validation_" + k, v.item())

            return loss

         def configure_optimizers(self):
            param_dicts = [
                  {"params": [p for n, p in self.named_parameters() if "backbone" not in n and p.requires_grad]},
                  {
                      "params": [p for n, p in self.named_parameters() if "backbone" in n and p.requires_grad],
                      "lr": self.lr_backbone,
                  },
            ]
            optimizer = torch.optim.AdamW(param_dicts, lr=self.lr,
                                      weight_decay=self.weight_decay)
            
            return optimizer

         def train_dataloader(self):
            return train_dataloader

         def val_dataloader(self):
            return val_dataloader


    model = Detr(lr=1e-4, lr_backbone=1e-5, weight_decay=1e-4, id2label=id2label, label2id=label2id)

    outputs = model(pixel_values=batch['pixel_values'], pixel_mask=batch['pixel_mask'])

    print(outputs.logits.shape)

    root_folder = "deformable_logs"

    # Define a logger with a custom directory
    logger = TensorBoardLogger(save_dir=root_folder, name='logs')

    trainer = Trainer(max_epochs=250, gradient_clip_val=0.1, logger=logger, default_root_dir=root_folder)
    trainer.fit(model)

    model.model.push_to_hub("jameszeng/deformable-detr-finetuned-pklot-full", private=True)
    processor.push_to_hub("jameszeng/deformable-detr-finetuned-pklot-full", private=True)

if __name__ == '__main__':
    try:
        set_start_method('spawn')
    except RuntimeError:
        pass

    main()
