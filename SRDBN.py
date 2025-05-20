import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk
import numpy as np
import matplotlib.pyplot as plt
import cv2
from deep_belief_network.dbn.tensorflow.models import SupervisedDBNRegression
from utils.data_processing import DataProcessing
from utils.image_file_util import *
from utils.scoring import *
from utils.visualizer import *

# --- Dummy Super-Resolution Methods ---
def nearest_neighbor(img):
    return cv2.resize(img, (img.shape[1] * 2, img.shape[0] * 2), interpolation=cv2.INTER_NEAREST)

def bilinear(img):
    return cv2.resize(img, (img.shape[1] * 2, img.shape[0] * 2), interpolation=cv2.INTER_LINEAR)

def bicubic(img):
    return cv2.resize(img, (img.shape[1] * 2, img.shape[0] * 2), interpolation=cv2.INTER_CUBIC)

def apply_dbn_super_resolution(img):

    dbn = SupervisedDBNRegression(
            hidden_layers_structure=[512,768,768,512],
            batch_size=512,
            learning_rate_rbm=0.01,
            n_epochs_rbm=500,
            activation_function='sigmoid',
            optimization_algorithm='sgd',
            learning_rate=0.001,
            n_iter_backprop=500,
        )
    
    dp = DataProcessing()

    dbn = dbn.load("model/model_mandrill.tif_f(brickwall_256.png)_512_x2_p16_s4_l[512, 768, 768, 512]_sigmoid_lr0.01_lrft0.001.h5")
    
    interpolated_test = dp.interpolate(img, 2)
    test_patches = dp.get_patches(interpolated_test)
    norm, _ = dp.normalize_for_rbm(test_patches)

    test_patches_flat = dp.preprocess_for_rbm(norm)
    norm=None

    result_flat = dbn.predict(test_patches_flat)
    result_flat = dp.proccess_output(test_patches_flat, result_flat)

    result_patches, _ = dp.inverse_preprocess(
        result_flat, (16, 16, 3)
    )

    reconstruct_image = dp.reconstruct_from_patches(
        result_patches, original_shape=(512,512,3))
    
    result_flat=None
    
    result_patches=None
    return reconstruct_image

METHODS = {
    "Nearest Neighbor": nearest_neighbor,
    "Bilinear": bilinear,
    "Bicubic": bicubic,
    "DBN": apply_dbn_super_resolution
}

class SuperResolutionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Super Resolution GUI")
        self.root.geometry("400x450")

        self.original_image = None
        self.result_images = {}  # method -> image

        self.btn_load = ttk.Button(root, text="Pilih Gambar Input", command=self.load_image)
        self.btn_load.pack(pady=5)

        self.btn_ref = ttk.Button(root, text="Pilih Gambar Referensi", command=self.load_reference_image)
        self.btn_ref.pack(pady=5)

        self.method_var = tk.StringVar()
        self.method_dropdown = ttk.Combobox(root, textvariable=self.method_var, values=list(METHODS.keys()), state="readonly")
        self.method_dropdown.pack(pady=5)
        self.method_dropdown.current(0)

        self.btn_run = ttk.Button(root, text="Terapkan Super Resolution", command=self.apply_method)
        self.btn_run.pack(pady=5)

        self.status_label = ttk.Label(root, text="Status: Menunggu input")
        self.status_label.pack(pady=5)

        self.compare_var1 = tk.StringVar()
        self.compare_var2 = tk.StringVar()

        ttk.Label(root, text="Bandingkan Gambar 1:").pack()
        self.compare_dropdown1 = ttk.Combobox(root, textvariable=self.compare_var1, state="readonly")
        self.compare_dropdown1.pack(pady=2)

        ttk.Label(root, text="Bandingkan Gambar 2:").pack()
        self.compare_dropdown2 = ttk.Combobox(root, textvariable=self.compare_var2, state="readonly")
        self.compare_dropdown2.pack(pady=2)

        self.btn_compare = ttk.Button(root, text="Bandingkan", command=self.compare_images)
        self.btn_compare.pack(pady=5)

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp")])
        if not file_path:
            return
        self.original_image = cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2RGB)
        messagebox.showinfo("Info", "Gambar input berhasil dimuat.")
        self.status_label.config(text="Status: Gambar input dimuat")

    def load_reference_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp")])
        if not file_path:
            return
        ref_image = cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2RGB)
        self.result_images["Referensi"] = ref_image
        self.update_compare_dropdowns()
        messagebox.showinfo("Info", "Gambar referensi berhasil dimuat.")
        self.status_label.config(text="Status: Gambar referensi dimuat")

    def apply_method(self):
        if self.original_image is None:
            messagebox.showerror("Error", "Silakan pilih gambar input terlebih dahulu.")
            return

        method_name = self.method_var.get()
        self.status_label.config(text=f"Status: Memproses dengan metode {method_name}...")
        self.root.update_idletasks()

        result = METHODS[method_name](self.original_image)
        self.result_images[method_name] = result
        self.update_compare_dropdowns()
        messagebox.showinfo("Info", f"Metode {method_name} telah diterapkan.")
        self.status_label.config(text=f"Status: Metode {method_name} selesai")

    def update_compare_dropdowns(self):
        choices = list(self.result_images.keys())
        self.compare_dropdown1['values'] = choices
        self.compare_dropdown2['values'] = choices
        if choices:
            self.compare_var1.set(choices[0])
            self.compare_var2.set(choices[-1])

    def compare_images(self):
        img1_name = self.compare_var1.get()
        img2_name = self.compare_var2.get()

        if not img1_name or not img2_name:
            messagebox.showerror("Error", "Pilih dua gambar untuk dibandingkan.")
            return

        img1 = self.result_images.get(img1_name)
        img2 = self.result_images.get(img2_name)

        if img1 is None or img2 is None:
            messagebox.showerror("Error", "Gambar tidak ditemukan.")
            return

        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].imshow(img1)
        axs[0].set_title(img1_name)
        axs[0].axis("off")
        axs[1].imshow(img2)
        axs[1].set_title(img2_name)
        axs[1].axis("off")
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    root = tk.Tk()
    app = SuperResolutionGUI(root)
    root.mainloop()
