import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
import numpy as np
import cv2

def load_image(file_path):
    # Muat gambar
    image_cv2 = cv2.imread(file_path)
    img = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)
    return img

def save_image_as(filename, buffer):
    # save gambar
    cv2.imwrite(filename, buffer)
    print(f"Image {filename} has been saved.")

def read_dicom_image(filepath):
    """
    Membaca file dicom dan mengembalikan pixel arraynya
    """
    # Baca file DICOM
    dicom_data = pydicom.read_file(filepath)
    dicom_image_array = None
    if 'WindowWidth' in dicom_image_array:
      print('Dataset has windowing')
      dicom_image_array = apply_voi_lut(dicom_data.pixel_array, dicom_data)
    else:
      # Ambil array gambar dari data DICOM
      dicom_image_array = dicom_data.pixel_array.astype(float)

    # Rescaling grey scale between 0-255
    image_2d_scaled = (np.maximum(dicom_image_array,0) / dicom_image_array.max()) * 255.0

    # Konversi ke format yang dapat diolah oleh OpenCV
    dicom_image_array = dicom_image_array.astype('uint8')

    return dicom_image_array


# Downscaling Image
def downscale_image(image, scale_factor):
    # Menggunakan resize dari OpenCV
    width = int(image.shape[1] * scale_factor)
    height = int(image.shape[0] * scale_factor)
    dim = (width, height)
    resized_image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    return resized_image