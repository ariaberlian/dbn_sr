from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import numpy as np

def calculate_psnr(original_image, enhanced_image):
    psnr_value = peak_signal_noise_ratio(original_image, enhanced_image)
    return psnr_value

def calculate_ssim(original_image, enhanced_image):
    # Inisialisasi variabel untuk menyimpan nilai SSIM untuk setiap channel
    ssim_values = []

    # Iterasi melalui setiap channel
    for channel in range(original_image.shape[2]):
        # Hitung SSIM untuk setiap channel
        ssim_channel, _ = structural_similarity(original_image[:, :, channel], enhanced_image[:, :, channel], full=True)

        # Tambahkan nilai SSIM ke dalam list
        ssim_values.append(ssim_channel)

    # Hitung rata-rata SSIM untuk semua channel
    average_ssim = np.mean(ssim_values)

    return average_ssim