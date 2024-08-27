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
    """ Load the PKLotSegmented dataset """

    transform = T.Compose([
        # flip the image horizontally with a probability of 0.5
        T.RandomHorizontalFlip(p=0.5),

        # resize the image to 224x224
        T.Resize((224,224)),

        # convert the image to a PyTorch
        T.ToTensor(),

        # normalize the image to the ImageNet mean and standard deviation
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    def target_transform(target):
        return torch.tensor(target['label'])

    dataset = PklotSegmentedDataset(img_dir='./datasets/pklot/PKLotSegmented/PUC', transform=transform, target_transform=target_transform)
    return dataset


def load_down_sampled_dataset():
    """ Load the down-sampled PKLotSegmented dataset """

    transform = T.Compose([
        # down sample to 32x32
        T.Resize((32,32)),

        # flip the image horizontally with a probability of 0.5
        T.RandomHorizontalFlip(p=0.5),

        # resize the image to 224x224
        T.Resize((224,224)), # then up sample to 224x224 (ImageNet size)

        # convert the image to a PyTorch
        T.ToTensor(),

        # normalize the image to the ImageNet mean and standard deviation
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    def target_transform(target):
        return torch.tensor(target['label'])

    dataset = PklotSegmentedDataset(img_dir='./datasets/pklot/PKLotSegmented/PUC', transform=transform, target_transform=target_transform)
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
    # Load the dataset
    dataset1= load_dataset()

    # Load the down-sampled dataset
    dataset2= load_down_sampled_dataset()

    # Create a composite dataset from the two datasets
    dataset = CompositeDataset(dataset1, dataset2)

    # Train the model
    train(dataset, ratio=0.8, num_epochs=30, num_classes=2)
