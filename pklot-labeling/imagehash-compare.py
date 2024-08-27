# This util is used to compare two images and calculate the similarity between them.
# Use histogram comparison to calculate the similarity between two images.

import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import glob
import imagehash
from PIL import Image

def normalized(image):

    # Load the image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

    # Convert image to float32 and scale to [0, 1]
    image = image.astype(np.float32) / 255.0

    # Define mean and std for each channel
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    # Normalize the image
    normalized_image = (image - mean) / std

    # To visualize the normalized image, you might want to convert it back to uint8
    show_normalized_image = np.clip(normalized_image* 255.0, 0, 255).astype(np.uint8)

    # Convert the image back to BGR
    show_normalized_image = cv2.cvtColor(show_normalized_image, cv2.COLOR_RGB2BGR)

    return normalized_image, show_normalized_image

def show_image_histograms(image1_file, image2_file):
    # Load images
    image1 = cv2.imread(image1_file)
    image2 = cv2.imread(image2_file)

    hash1 = imagehash.average_hash(Image.open(image1_file))
    hash2 = imagehash.average_hash(Image.open(image2_file))
    phashSimilarity = hash1 - hash2

    print('Hash1:', hash1)
    print('Hash2:', hash2)
    print('Hash similarity:', phashSimilarity)

    # normalize the images using mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    normalized_image1, visual_image1 = normalized(image1)
    normalized_image2, visual_image2 = normalized(image2)

    # Calculate histograms for each channel
    colors = ('b', 'g', 'r')
    hist1 = {}
    hist2 = {}

    for i, color in enumerate(colors):
        hist1[color] = cv2.calcHist([image1], [i], None, [256], [0, 256])
        hist2[color] = cv2.calcHist([image2], [i], None, [256], [0, 256])

    normalized_hist1 = {}
    normalized_hist2 = {}
    for i, color in enumerate(colors):
        normalized_hist1[color] = cv2.calcHist([visual_image1], [i], None, [256], [0, 256])
        normalized_hist2[color] = cv2.calcHist([visual_image2], [i], None, [256], [0, 256])

    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # calculate the histograms
    gray_hist1 = cv2.calcHist([gray1], [0], None, [256], [0, 256])
    gray_hist2 = cv2.calcHist([gray2], [0], None, [256], [0, 256])

    gray_visual1 = cv2.cvtColor(visual_image1, cv2.COLOR_BGR2GRAY)
    gray_visual2 = cv2.cvtColor(visual_image2, cv2.COLOR_BGR2GRAY)

    # calculate the histograms
    gray_visual_histo1 = cv2.calcHist([gray_visual1], [0], None, [256], [0, 256])
    gray_visual_histo2 = cv2.calcHist([gray_visual2], [0], None, [256], [0, 256])


    # Plotting
    plt.figure(figsize=(20, 10))

    # Plot histograms for image1
    plt.subplot(2, 4, 1)
    for color in colors:
        plt.plot(hist1[color], color=color)
    plt.plot(gray_hist1, color='black')
    plt.title('Histogram of Image 1')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')

    # Plot normalized histograms for image1
    plt.subplot(2, 4, 2)
    for color in colors:
        plt.plot(normalized_hist1[color], color=color)
    plt.title('Norm Histogram of Image 1')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')

    # Plot histograms for image2
    plt.subplot(2, 4, 3)
    for color in colors:
        plt.plot(hist2[color], color=color)
    plt.plot(gray_hist2, color='black')
    plt.title('Histogram of Image 2')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')

    # Plot normalized histograms for image2
    plt.subplot(2, 4, 4)
    for color in colors:
        plt.plot(normalized_hist2[color], color=color)
    plt.title('Norm Histogram of Image 2')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')


    # Plot image1
    plt.subplot(2, 4, 5)
    image1_rgb = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    plt.imshow(image1_rgb)
    plt.title('Image 1')
    plt.axis('off')

    # Plot norm image1
    plt.subplot(2, 4, 6)
    plt.imshow(cv2.cvtColor(visual_image1, cv2.COLOR_BGR2RGB))
    plt.title('Normalized Image 1')
    plt.axis('off')

    # Plot image2
    plt.subplot(2, 4, 7)
    image2_rgb = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
    plt.imshow(image2_rgb)
    plt.title('Image 2')
    plt.axis('off')

    # Plot norm image2
    plt.subplot(2, 4, 8)
    plt.imshow(cv2.cvtColor(visual_image2, cv2.COLOR_BGR2RGB))
    plt.title('Normalized Image 2')
    plt.axis('off')

    cv2.normalize(gray_hist1, gray_hist1)
    cv2.normalize(gray_hist2, gray_hist2)

    similarity1 = cv2.compareHist(gray_hist1, gray_hist2, cv2.HISTCMP_CORREL)
    print('Similarity1:', similarity1)

    similarity2 = cv2.compareHist(gray_visual_histo1, gray_visual_histo2, cv2.HISTCMP_CORREL)
    print('Similarity2:', similarity2)

    plt.suptitle(f'Similarity: gray={similarity1:.2f} normalized={similarity2:.2f} phashSimilarity={phashSimilarity}')

    # Show plots
    plt.tight_layout()
    plt.show(block=True)


def main():
    # program arguments
    parser = argparse.ArgumentParser(description='Image similarity comparison')
    parser.add_argument('imagedir', help='path to the image directory')
    args = parser.parse_args()

    # load all images in the directory, extension should be .jpg
    images = glob.glob(os.path.join(args.imagedir,'*.jpg'))

    # compare all images
    for i in range(len(images)):
        for j in range(i+1, len(images)):
            show_image_histograms(os.path.join(args.imagedir, images[i]), os.path.join(args.imagedir, images[j]))

if __name__ == '__main__':
    main()