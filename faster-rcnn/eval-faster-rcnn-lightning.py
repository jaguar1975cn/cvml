import torch
import torchvision
import os
import matplotlib.pyplot as plt
import torchvision.transforms as T
from torch.utils.data import DataLoader
from coco_eval import CocoEvaluator
from torchvision.datasets import CocoDetection
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tqdm import tqdm
from huggingface_hub import hf_hub_download
from multiprocessing import set_start_method

try:
    set_start_method('spawn')
except RuntimeError:
    pass

# set the device
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
            target_dict["image_id"] = -1
        else:
            target_dict["boxes"] = torch.tensor([ [t['bbox'][0], t['bbox'][1], t['bbox'][0] + t['bbox'][2], t['bbox'][1] + t['bbox'][3] ] for t in target], dtype=torch.float32).to(device)
            target_dict["labels"] = torch.tensor([t['category_id'] for t in target]).to(device)
            target_dict["image_id"] = target[0]["image_id"]

        targets.append(target_dict)

    images = torch.stack(images, dim=0)

    return images, targets

# load dataset
def load_data(root, ann_file, batch_size=4, num_workers=1):
    test_dataset = CocoDetection(root,   os.path.join(root, ann_file), T.ToTensor())
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=num_workers)
    return (test_dataset, test_data_loader)

# download model and save it to local cache
def download_model(repo, model_name):
    """ Download the model from huggingface hub, the return value is the model file path """
    return hf_hub_download(repo_id=repo, filename=model_name)
    return model_name

# load model
def load_model(model_file):

    # create the model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)

    # Replace the classifier with a new one that has the correct number of output classes
    num_classes = 3  # Replace with the number of classes in your dataset
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.roi_heads.detections_per_img = 300

    # move it to device
    model.to(device)

    # set the model to evaluation mode
    model.eval()

    # load the trained model
    model.load_state_dict(torch.load(model_file, map_location=device))
    return model

# show the image with the bounding boxes and target boxes
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


# detect the image
def detect(index, model, img:torch.Tensor, target):

    # make it 3 channels and 1 batch
    batch = img.unsqueeze(0)

    # Run the image through the model
    # disable gradient calculation
    with torch.no_grad():
        output = model(batch)

    # Print the predicted classes and bounding boxes
    output = output[0]

    #show(img, output, target)

    # find count of each class
    count = torch.bincount(output['labels'])

    if len(count)<=1:
        print("No object detected")
        if len(target['boxes'])==0:
            # no ground truth, the AP is undefined
            return {}

        # no object detected, but there are ground truth, the AP is 0
        return {}

    # class_labels = {0: 'spaces', 1: 'space-empty', 2: 'space-occupied'}

    if len(count)==3:
        print("{}) Image #{}: found {} spaces and {} cars".format(index, target["image_id"], count[1], count[2]))

    if target["image_id"] == -1:
        return None

    res = {target["image_id"]: output}

    return res

def main():

    repo_id = 'jameszeng/faster-rcnn-finetuned-pklot-full'
    model_file = 'fasterrcnn_resnet50_fpn.pth'
    img_folder = 'datasets/pklot/images/test'
    ann_file = 'full.json'
    batch_size = 1
    num_workers = 1

    # download the model from huggingface hub, the output is the model file name
    model_path = download_model(repo_id, model_file)

    # load the model
    model = load_model(model_path)

    # load test dataset
    test_dataset, test_dataloader = load_data(img_folder, ann_file, batch_size, num_workers)

    print("Image folder:", img_folder)
    print("Number of test examples:", len(test_dataloader))
    print("Batch size:", batch_size)

    # initialize coco evaluator
    coco_evaluator = CocoEvaluator(test_dataset.coco, iou_types=["bbox"], maxDets=300)

    for index, (imgs, target) in enumerate(tqdm(test_dataloader)):
        img = imgs[0]

        # detect objects in the image
        batch_dt = detect(index, model, img, target[0])

        # update the coco evaluator
        if batch_dt:
            coco_evaluator.update(batch_dt)

        # if index > 3:
        #     break

    # print summaries
    coco_evaluator.synchronize_between_processes()
    coco_evaluator.accumulate()
    coco_evaluator.summarize()

if __name__ == '__main__':
    main()
