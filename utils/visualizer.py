import numpy as np
import cv2
from matplotlib import gridspec
import matplotlib.pyplot as plt

def visualize4image(image1, title1, image2, title2, image3, title3, image4, title4):
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    axes[0, 0].imshow(image1)
    axes[0, 0].axis('off')  
    axes[0, 0].set_title(title1)

    axes[0, 1].imshow(image2)
    axes[0, 1].axis('off')
    axes[0, 1].set_title(title2)

    axes[1, 0].imshow(image3)
    axes[1, 0].axis('off')
    axes[1, 0].set_title(title3)

    axes[1, 1].imshow(image4)
    axes[1, 1].axis('off')
    axes[1, 1].set_title(title4)

    plt.tight_layout()
    plt.ion()
    plt.show()
    plt.pause(0.001)

def visualize4Histogram(image1, title1, image2, title2, image3, title3, image4, title4):
    plt.figure(figsize=(12, 10))

    plt.subplot(2, 2, 1)
    plt.hist(image1.flatten(), bins=256, range=(0, 256), color='b', alpha=0.5)
    plt.title('Histogram of ' + title1)
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')

    plt.subplot(2, 2, 2)
    plt.hist(image2.flatten(), bins=256, range=(0, 256), color='r', alpha=0.5)
    plt.title('Histogram of ' + title2)
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')

    # Histogram for Normalized Original Image
    plt.subplot(2, 2, 3)
    plt.hist(image3.flatten(), bins=256, range=(0, 256), color='g', alpha=0.5)
    plt.title('Histogram of ' + title3)
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')

    # Histogram for Normalized Reconstructed Image
    plt.subplot(2, 2, 4)
    plt.hist(image4.flatten(), bins=256, range=(0, 256), color='orange', alpha=0.5)
    plt.title('Histogram of ' + title4)
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')

    # Adjust layout
    plt.tight_layout()
    plt.ion()
    plt.show()
    plt.pause(0.001)


def visualize_histogram_compare(original_image, reconstruct_image, title1='Original Image', title2='Reconstructed Image'):
    plt.figure(figsize=(12, 6))

    # Histogram Sebelum Normalisasi
    plt.subplot(2, 2, 1)
    plt.hist(original_image.flatten(), bins=256, range=(0, 256), color='b', alpha=0.5)
    plt.title('Histogram '+title1)
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')

    # Histogram Sesudah Normalisasi
    plt.subplot(2, 2, 2)
    plt.hist(reconstruct_image.flatten(), bins=256, range=(0, 256), color='r', alpha=0.5)
    plt.title('Histogram '+title2)
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')

    # Plot Gambar Asli
    plt.subplot(2, 2, 3)
    plt.imshow(original_image, cmap='gray')
    plt.title(title1)

    # Plot Hasil Normalisasi
    plt.subplot(2, 2, 4)
    plt.imshow(reconstruct_image, cmap='gray')
    plt.title(title2)

    plt.tight_layout()
    plt.ion()
    plt.show()
    plt.pause(0.001)

def visualize_histogram(array, title, range=(0, 1), xlabel='Value', ylabel='Frequency'):
    plt.hist(array.flatten(), bins=256, range=range, color='r', alpha=0.5)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.ion()
    plt.show()
    plt.pause(0.001)

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
    plt.ion()
    plt.show()
    plt.pause(0.001)

def visualize_image(image, title):
    plt.imshow(image)
    plt.title(title)
    plt.ion()
    plt.show()
    plt.pause(0.001)

def visualize_dct(dct_image, title=""):
    # Membuat figure untuk menampung subplots
    plt.figure(figsize=(15, 5))  # Lebar lebih besar agar saluran bisa ditampilkan bersebelahan

    # Loop untuk setiap saluran
    for i, channel in enumerate(['Red', 'Green', 'Blue']):
        # Mengambil magnitudo dari DCT dan menormalisasikannya
        dct_image_magnitude = np.log(np.abs(dct_image[:, :, i]) + 1)  # Menghindari log(0)
        
        # Membuat subplot untuk setiap saluran
        plt.subplot(1, 3, i + 1)  # 1 baris, 3 kolom
        plt.imshow(dct_image_magnitude, cmap='gray')
        plt.title(f'{title}: DCT Magnitude in Log Domain - {channel}')
        plt.colorbar()
        plt.axis('off')

    plt.tight_layout()  # Mengatur layout agar tidak tumpang tindih
    plt.ion()
    plt.show()
    plt.pause(0.001)