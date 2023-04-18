import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torchvision.transforms as T

from PIL import Image

img = Image.open('test.png')

convert_tensor = T.ToTensor()

tensor=convert_tensor(img)
batch = tensor.unsqueeze(0)

# make it 3 channels and 1 batch
prewitt = torch.tensor([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=torch.float32).repeat(1,3,1,1)
roberts = torch.tensor([[[1, 0], [0, -1]]], dtype=torch.float32).repeat(1,3,1,1)
sobel = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32).repeat(1,3,1,1)
# sobel 5x5
sobel5 = torch.tensor(
    [
    [-1, -2, 0, 2, 1],
    [-2, -3, 0, 3, 2],
    [-3, -5, 0, 5, 3],
    [-2, -3, 0, 3, 2],
    [-1, -2, 0, 2, 1]
    ], dtype=torch.float32).repeat(1,3,1,1)


# add all filters into a list
filters = [prewitt, roberts, sobel, sobel5]

fig, ax = plt.subplots(1, 5)

# show original image
#ax.imshow(t.permute(1, 2, 0) )
ax[0].imshow(torch.einsum('cwh->whc', tensor) )

for i, f in enumerate(filters):
    # apply filter to tensor
    filtered_tensor = F.conv2d(batch, f, padding=0)
    ax[i+1].imshow(torch.einsum('cwh->whc', filtered_tensor[0]) )

plt.show()

