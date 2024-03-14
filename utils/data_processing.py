import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy.fftpack import dct, idct
import cv2

class DataProcessing:
    def __init__(self):
        # Inisialisasi dan fit-transform dengan MinMaxScaler
        self.scaler = MinMaxScaler()

    def interpolate(self, original_image, interpolation_factor):
        """
        Interpolasi bicubic gambar dengan skala interpolation_factor
        """
        # Interpolasi untuk memperbesar gambar
        enlarged_image = cv2.resize(original_image, None, fx=interpolation_factor, fy=interpolation_factor, interpolation=cv2.INTER_CUBIC)

        return enlarged_image

    def dct_transform(self, image):
        # Convert image to float64 for DCT computation
        image_float64 = image.astype(np.float64)

        # Apply 2D DCT on each channel
        dct_result = np.zeros_like(image_float64)
        for channel in range(image.shape[2]):
            dct_result[:, :, channel] = dct(dct(image_float64[:, :, channel], axis=0, norm='ortho'), axis=1, norm='ortho')

        return dct_result

    def idct_transform(self, dct_result):
        # Apply inverse 2D DCT on each channel
        image_restored = np.zeros_like(dct_result)
        for channel in range(dct_result.shape[2]):
            image_restored[:, :, channel] = idct(idct(dct_result[:, :, channel], axis=0, norm='ortho'), axis=1, norm='ortho')

        # Convert the restored image back to uint8
        image_restored = np.clip(image_restored, 0, 255)
        image_restored = image_restored.astype(np.uint8)

        return image_restored

    def _normalize_for_rbm(self, patches):
        """
        Normalisasi gambar untuk dapat dimasukkan ke dalam Restricted Boltzmann Machine (RBM).
        """
        # Perform DCT transform'
        dct_results_list = []
        for patch in patches:
            dct_result = self.dct_transform(patch)
            dct_results_list.append(dct_result)

        dct_results_list = np.array(dct_results_list, dtype=np.float64)
        dct_results_list = dct_results_list.reshape(patches.shape)

        print("DCT coefficients shape: ", dct_results_list.shape)
        # print("This is after DCT: ")
        # print(dct_results_list[0])
        # print("After dct max:", np.max(dct_results_list))
        # print("After dct min:", np.min(dct_results_list))


        # Flatten DCT coefficients for normalization
        flattened_dct_coefficients = dct_results_list.reshape(
            patches.shape[0], -1)

        # print("flaten shape: ", flattened_dct_coefficients.shape)

        # Fit-transform using the same scaler instance for all images
        normalized_data = self.scaler.fit_transform(flattened_dct_coefficients)

        # Reshape back to the original form after normalization
        normalized_dct_coefficients = normalized_data.reshape(
            dct_results_list.shape)

        # print("Normalized DCT Coefficients shape: ", normalized_dct_coefficients.shape)
        # print("This is after normalization: ", normalized_dct_coefficients[0])
        # Save minimum and maximum values before normalization

        return normalized_dct_coefficients

    def _zigzag(self, input):
        # initializing the variables
        # ----------------------------------
        h = 0
        v = 0

        vmin = 0
        hmin = 0

        vmax = input.shape[0]
        hmax = input.shape[1]
        channel = input.shape[2]

        # print(vmax ,hmax, channel )

        i = 0

        output = np.zeros((vmax * hmax * channel))
        # ----------------------------------

        while ((v < vmax) and (h < hmax)):
            # print(output)
            # print(f"{v}, {h}")
            if ((h + v) % 2) == 0:                 # going up
                if (v == vmin):
                    # print(1)
                    for c in range(channel):
                        # if we got to the first line
                        output[i+c] = input[v, h, c]
                    i = i + channel - 1
                    if (h == hmax-1):
                        # print("chihuy")
                        v = v + 1
                    else:
                        h = h + 1

                    i = i + 1

                elif ((h == hmax - 1) and (v < vmax)):   # if we got to the last column
                    # print(2)
                    for c in range(channel):
                        output[i+c] = input[v, h, c]
                    i = i + channel - 1
                    v = v + 1
                    i = i + 1

                elif ((v > vmin) and (h < hmax - 1)):    # all other cases
                    # print(3)
                    for c in range(channel):
                        output[i+c] = input[v, h, c]
                    i = i + channel - 1
                    v = v - 1
                    h = h + 1
                    i = i + 1
            else:                                    # going down

                if ((v == vmax - 1) and (h <= hmax - 1)):       # if we got to the last line
                    # print(4)
                    for c in range(channel):
                        output[i+c] = input[v, h, c]
                    i = i + channel - 1
                    h = h + 1
                    i = i + 1

                elif (h == hmin):                  # if we got to the first column
                    # print(5)
                    for c in range(channel):
                        output[i+c] = input[v, h, c]
                    i = i + channel - 1

                    if (v == vmax - 1):
                        h = h + 1
                    else:
                        v = v + 1

                    i = i + 1

                elif ((v < vmax - 1) and (h > hmin)):     # all other cases
                    # print(6)
                    for c in range(channel):
                        output[i+c] = input[v, h, c]
                    i = i + channel - 1
                    v = v + 1
                    h = h - 1
                    i = i + 1

            if ((v == vmax-1) and (h == hmax-1)):          # bottom right element
                # print(7)
                for c in range(channel):
                    output[i+c] = input[v, h, c]
                i = i + channel - 1
                break

        # print ('v:',v,', h:',h,', i:',i)
        return output

    # Inverse zigzag scan of a matrix
    # Arguments are: a 1-by-m*n array,
    # where m & n are vertical & horizontal sizes of an output matrix.
    # Function returns a two-dimensional matrix of defined sizes,
    # consisting of input array items gathered by a zigzag method.
    #
    # Matlab Code:
    # Alexey S. Sokolov a.k.a. nICKEL, Moscow, Russia
    # June 2007
    # alex.nickel@gmail.com

    def _inverse_zigzag(self, input, original_shape: tuple):

        # print input.shape

        # initializing the variables
        # ----------------------------------
        h = 0
        v = 0

        vmin = 0
        hmin = 0

        vmax, hmax, channel = original_shape

        output = np.zeros((vmax, hmax, channel))

        i = 0
        # ----------------------------------

        while ((v < vmax) and (h < hmax)):
            # print ('v:',v,', h:',h,', i:',i)
            if ((h + v) % 2) == 0:                 # going up

                if (v == vmin):
                    # print(1)
                    for c in range(channel):
                        # if we got to the first line
                        output[v, h, c] = input[i+c]
                    i = i + channel-1
                    if (h == hmax-1):
                        v = v + 1
                    else:
                        h = h + 1

                    i = i + 1

                elif ((h == hmax - 1) and (v < vmax)):   # if we got to the last column
                    # print(2)
                    for c in range(channel):
                        # if we got to the first line
                        output[v, h, c] = input[i+c]
                    i = i + channel-1
                    v = v + 1
                    i = i + 1

                elif ((v > vmin) and (h < hmax - 1)):    # all other cases
                    # print(3)
                    for c in range(channel):
                        # if we got to the first line
                        output[v, h, c] = input[i+c]
                    i = i + channel-1
                    v = v - 1
                    h = h + 1
                    i = i + 1

            else:                                    # going down

                if ((v == vmax - 1) and (h <= hmax - 1)):       # if we got to the last line
                    # print(4)
                    for c in range(channel):
                        # if we got to the first line
                        output[v, h, c] = input[i+c]
                    i = i + channel-1
                    h = h + 1
                    i = i + 1

                elif (h == hmin):                  # if we got to the first column
                    # print(5)
                    for c in range(channel):
                        # if we got to the first line
                        output[v, h, c] = input[i+c]
                    i = i + channel-1
                    if (v == vmax - 1):
                        h = h + 1
                    else:
                        v = v + 1
                    i = i + 1

                elif ((v < vmax - 1) and (h > hmin)):     # all other cases
                    for c in range(channel):
                        output[v, h, c] = input[i+c]
                    i = i + channel-1
                    v = v + 1
                    h = h - 1
                    i = i + 1

            if ((v == vmax-1) and (h == hmax-1)):          # bottom right element
                # print(7)
                for c in range(channel):
                    # if we got to the first line
                    output[v, h, c] = input[i+c]
                i = i + channel-1
                break

        return output

    def preprocess_for_rbm(self, images):
        """
        Persiapkan data untuk dimasukkan ke dalam RBM.
        Image di normalisasi dengan dct sehingga dalam domain frekuensi dengan rentang [0,1]

        Input: patch images
        Output: flatten arrays: array-like (Number of patch images, (patch.shape[0] * patch.shape[1] * patch.shape[2]))
        """

        # Normalisasi gambar untuk RBM
        normalized_coefficients = self._normalize_for_rbm(images)

        # Flattening hasil untuk masukan RBM
        flattened_dct_coefficients = [self._zigzag(
            coeff) for coeff in normalized_coefficients]
        flattened_dct_coefficients = np.asarray(
            flattened_dct_coefficients, dtype=np.float64)

        # print("Ready for rbm array shape: ", flattened_dct_coefficients.shape)
        # print("This is after zigzag:")
        # print(flattened_dct_coefficients[0])
        return flattened_dct_coefficients

    def inverse_preprocess(self, coefficients, original_patch_shape):
        """
        Mengembalikan inverse koefisien DCT yang dinormalisasi ke gambar asli dengan inverse langkah zigzag

        Input: coefficients: array-like (Number of patch images, (patch.shape[0] * patch.shape[1] * patch.shape[2]))
        """

        # Reshape ke bentuk yang sesuai dengan gambar asli
        reshaped_coefficients = [self._inverse_zigzag(
            coef, original_patch_shape) for coef in coefficients]
        reshaped_coefficients = np.asarray(
            reshaped_coefficients, dtype=np.float64)

        # print("After inverse zigzag shape: ", reshaped_coefficients.shape)
        # print("this is after inverse zigzag:")
        # print(reshaped_coefficients[0])

        # Denormalisasi nilai koefisien
        denormalized_coefficients = self.scaler.inverse_transform(
            reshaped_coefficients.reshape(reshaped_coefficients.shape[0], -1))

        # Reshape kembali ke bentuk asli setelah normalisasi
        denormalized_coefficients = denormalized_coefficients.reshape(
            reshaped_coefficients.shape)

        print("Denormalized coeffients array shape: ", denormalized_coefficients.shape)
        # print("This is after denormalize:")
        # # print(denormalized_coefficients[0])
        # print("before idct max:", np.max(denormalized_coefficients))
        # print("before idct min:", np.min(denormalized_coefficients))

        # Perform inverse DCT transform
        restored_images_list = []
        for res in denormalized_coefficients:
            restored_image = self.idct_transform(res)
            restored_images_list.append(restored_image)

        restored_images_list = np.array(restored_images_list, dtype=np.uint8)
        restored_images_list = restored_images_list.reshape(denormalized_coefficients.shape)

        return restored_images_list

    def get_patches(self, input_image: np.ndarray, patch_size: tuple = (16, 16), stride: tuple = (4, 4)):
        """
        Ambil patch dari input_image dengan ukuran patch_size dan langkah sejauh stride
        """
        patches = []

        i = j = 0
        height, width, channels = input_image.shape[0], input_image.shape[1], input_image.shape[2]

        while i + patch_size[0] <= height:
            j = 0
            while j + patch_size[1] <= width:
                if i + patch_size[0] <= height and j + patch_size[1] <= width:
                    patches.append(
                        input_image[i:i+patch_size[0], j:j+patch_size[1], :])
                j += stride[1]
            i += stride[0]

        print("Jumlah Patches = ", len(patches))
        patches = np.asarray(patches).reshape(-1,
                                              patch_size[0], patch_size[1], channels)
        # visualize_patches(patches, (8,8))
        return np.asarray(patches).reshape(-1, patch_size[0], patch_size[1], channels)

    def reconstruct_from_patches(self, patches, original_shape, patch_size=(16, 16), stride=(4, 4)):
        """
        Rekonstruksi gambar utuh dari patches.
        """
        height, width, channels = original_shape
        num_rows = (height - patch_size[0]) // stride[0] + 1
        num_cols = (width - patch_size[1]) // stride[1] + 1

        reconstructed_image = np.zeros(
            (height, width, channels), dtype=np.float64)
        count_matrix = np.zeros((height, width, channels), dtype=np.float64)

        patch_idx = 0
        for i in range(num_rows):
            for j in range(num_cols):
                start_row, start_col = i * stride[0], j * stride[1]
                end_row, end_col = start_row + \
                    patch_size[0], start_col + patch_size[1]
                reconstructed_image[start_row:end_row,
                                    start_col:end_col, :] += patches[patch_idx]
                count_matrix[start_row:end_row, start_col:end_col, :] += 1
                patch_idx += 1

        # Normalisasi dengan membagi dengan jumlah kontribusi setiap patch
        reconstructed_image /= count_matrix

        # Pastikan tipe data akhir sesuai dengan kebutuhan
        return reconstructed_image.astype(np.uint8)

    def proccess_output(self, u, r, beta, s):
        I = u.shape[1]
        high_freq_start = I//s
        for i in range(high_freq_start, I):
            u[: ,i] = beta*r[:, i]
        return u