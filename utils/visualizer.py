import numpy as np
import cv2
from matplotlib import gridspec
import matplotlib.pyplot as plt


def visualize_histogram_compare(img1, img2, title1='Original Image', title2='Reconstructed Image'):
    plt.figure(figsize=(12, 6))

    # Histogram Sebelum Normalisasi
    plt.subplot(2, 2, 1)
    plt.hist(img1.flatten(), bins=256, range=(0, 256), color='b', alpha=0.5)
    plt.title("Histogram "+title1)
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')

    # Histogram Sesudah Normalisasi
    plt.subplot(2, 2, 2)
    plt.hist(img2.flatten(), bins=256, range=(0, 256), color='r', alpha=0.5)
    plt.title("Histogram "+title2)
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')

    # Plot Gambar Asli
    plt.subplot(2, 2, 3)
    plt.imshow(img1, cmap='gray')
    plt.title('Original Image')

    # Plot Hasil Normalisasi
    plt.subplot(2, 2, 4)
    plt.imshow(img2, cmap='gray')
    plt.title('Reconstructed Image')

    plt.tight_layout()
    plt.show()

def visualize_histogram(array, title, range=(0, 1), xlabel='Value', ylabel='Frequency'):
    plt.hist(array.flatten(), bins=256, range=range, color='r', alpha=0.5)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

def visualize_patches(patches, title, visualize_size: tuple = (6,6)):
    # visualize patches example
    fig = plt.figure(figsize=(patches.shape[1], patches.shape[2]))
    plt.suptitle(title, fontsize=16)  # Menambahkan judul
    grid = gridspec.GridSpec(visualize_size[0], visualize_size[1], wspace=0.2, hspace=0.2)

    for i in range(visualize_size[0]):
        for j in range(visualize_size[1]):
            # Mengambil patch image
            patch = patches[i * visualize_size[0] + j]

            # Menampilkan patch image tanpa skala
            ax = plt.subplot(grid[i, j])
            ax.imshow(patch)
            ax.axis('off')  # Menyembunyikan sumbu

    plt.show()

def visualize_image(image, title):
    plt.imshow(image)
    plt.title(title)
    plt.show()

# Visualize DCT coefficients in the log domain
def visualize_dct(dct_coefficients, block_size, title):

    # Take the logarithm of the absolute values of DCT coefficients
    log_dct = np.log(np.abs(dct_coefficients))

    # Display the checkboard pattern in the log domain
    plt.imshow(log_dct, cmap='gray')
    plt.title(f'Checkboard Pattern in Log Domain of {title}')
    plt.colorbar()
    plt.show()