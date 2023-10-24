from transformers import DeformableDetrImageProcessor, DeformableDetrForObjectDetection
import torch
import torchvision
import os
import numpy
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torchvision.transforms as T


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

class_labels = {0: 'spaces', 1: 'space-empty', 2: 'space-occupied'}
model_name = "jameszeng/deformable-detr-finetuned-pklot-full"
model = DeformableDetrForObjectDetection.from_pretrained(model_name, id2label=class_labels)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
processor = DeformableDetrImageProcessor.from_pretrained(model_name)
def collate_fn(batch):
    pixel_values = [item[0] for item in batch]
    encoding = processor.pad(pixel_values, return_tensors="pt")
    labels = [item[1] for item in batch]
    batch = {}
    batch['pixel_values'] = encoding['pixel_values']
    batch['pixel_mask'] = encoding['pixel_mask']
    batch['labels'] = labels
    return batch

img_folder = '../datasets/pklot/images/test'
batch_size = 4
val_dataset = CocoDetection(img_folder=img_folder, processor=processor)
val_dataloader = DataLoader(val_dataset, collate_fn=collate_fn, batch_size=batch_size)

print("Image folder:", img_folder)
print("Number of test examples:", len(val_dataset))
print("Batch size:", batch_size)

def convert_to_xywh(boxes):
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)

def prepare_for_coco_detection(predictions):
    coco_results = []
    for original_id, prediction in predictions.items():
        if len(prediction) == 0:
            continue

        boxes = prediction["boxes"]
        boxes = convert_to_xywh(boxes).tolist()
        scores = prediction["scores"].tolist()
        labels = prediction["labels"].tolist()

        coco_results.extend(
            [
                {
                    "image_id": original_id,
                    "category_id": labels[k],
                    "bbox": box,
                    "score": scores[k],
                }
                for k, box in enumerate(boxes)
            ]
        )
    return coco_results


def show(prediction, image):
    """ Plot an image with bounding boxes """
    fig, ax = plt.subplots()
    ax.imshow(image)
    index = 0
    for bbox in prediction['boxes']:
        if prediction['labels'][index] == 1:
            color = 'g'
        elif prediction['labels'][index] == 2:
            color = 'r'
        rect = plt.Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1], linewidth=1, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        index = index + 1
    plt.show()


from coco_eval import CocoEvaluator
from tqdm import tqdm

import numpy as np

# initialize evaluator with ground truth (gt)
evaluator = CocoEvaluator(coco_gt=val_dataset.coco, iou_types=["bbox"])
evaluator.coco_eval["bbox"].params.maxDets=[1,10,300]

print("Running evaluation...")
for idx, batch in enumerate(tqdm(val_dataloader)):
    # get the inputs
    pixel_values = batch["pixel_values"].to(device)
    pixel_mask = batch["pixel_mask"].to(device)
    labels = [{k: v.to(device) for k, v in t.items()} for t in batch["labels"]] # these are in DETR format, resized + normalized

    # forward pass
    with torch.no_grad():
      outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)

    # turn into a list of dictionaries (one item for each example in the batch)
    orig_target_sizes = torch.stack([target["orig_size"] for target in labels], dim=0)
    results = processor.post_process_object_detection(outputs, target_sizes=orig_target_sizes, top_k=300)

    # provide to metric
    # metric expects a list of dictionaries, each item
    # containing image_id, category_id, bbox and score keys
    predictions = {target['image_id'].item(): output for target, output in zip(labels, results)}

    # print the image_id, number of predictions and the class counts
    for index, (image_id, prediction) in enumerate(predictions.items()):
        labels = prediction['labels']
        unique_labels, counts = torch.unique(labels, return_counts=True)
        label_counts_dict = {class_labels[label.item()]: count.item() for label, count in zip(unique_labels, counts)}
        print("image", image_id, "total", len(labels), label_counts_dict)
        # show(prediction, val_dataset._load_image(image_id))

    predictions = prepare_for_coco_detection(predictions)
    evaluator.update(predictions)

evaluator.synchronize_between_processes()
evaluator.accumulate()
evaluator.summarize()
