import torch
from dataset import PklotSegmentedDataset
from dataset import CompositeDataset
from train import train
import torchvision.transforms as T
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from multiprocessing import set_start_method

try:
    set_start_method('spawn')
except RuntimeError:
    pass


def load_dataset():
    # load the dataset
    transform = T.Compose([
        T.Resize((224,224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    def target_transfor(target):
        return torch.tensor(target['label'])

    dataset = PklotSegmentedDataset(img_dir='./datasets/pklot/PKLotSegmented/PUC', transform=transform, target_transform=target_transfor)
    return dataset

def load_down_sampled_dataset():
    # load the dataset
    transform = T.Compose([
        T.Resize((24,24)), # down sample to 24x24
        T.Resize((224,224)), # then up sample to 224x224 (ImageNet size)
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    def target_transfor(target):
        return torch.tensor(target['label'])

    dataset = PklotSegmentedDataset(img_dir='./datasets/pklot/PKLotSegmented/PUC', transform=transform, target_transform=target_transfor)
    return dataset

def show_data(dataset):
    data_loader = DataLoader(dataset, batch_size=10, shuffle=True)
    transform = T.ToPILImage()
    it = iter(data_loader)
    figure = plt.figure(figsize=(8, 8))
    cols, rows = 3, 3
    images, labels = next(it)
    for i in range(1, cols * rows + 1):
        figure.add_subplot(rows, cols, i)
        plt.title(labels['class'][i] + " - " + labels['weather'][i] )
        plt.axis("off")
        plt.imshow(transform(images[i]))
    plt.show()


if __name__ == '__main__':
    dataset1= load_dataset()
    dataset2= load_down_sampled_dataset()

    dataset = CompositeDataset(dataset1, dataset2)

    train(dataset, 0.8, 30, 2)
