import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk


class PixelPickerZoomApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Pixel Picker Zoom")

        # UI Elements
        self.upload_btn = tk.Button(root, text="Upload Image", command=self.upload_image)
        self.upload_btn.pack()

        self.coord_label = tk.Label(root, text="Enter pixel coordinates (x, y):")
        self.coord_label.pack()

        self.coord_entry = tk.Entry(root)
        self.coord_entry.pack()

        self.zoom_btn = tk.Button(root, text="Zoom", command=self.zoom_image)
        self.zoom_btn.pack()

        self.canvas_original = tk.Canvas(root, width=400, height=400, bg="gray")
        self.canvas_original.pack(side=tk.LEFT, padx=10, pady=10)

        self.canvas_zoom = tk.Canvas(root, width=400, height=400, bg="gray")
        self.canvas_zoom.pack(side=tk.RIGHT, padx=10, pady=10)

        # Image attributes
        self.image = None
        self.image_tk = None
        self.zoomed_image = None
        self.zoomed_image_tk = None

    def upload_image(self):
        filepath = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.tif")])
        if not filepath:
            return

        try:
            self.image = Image.open(filepath)
            self.image_tk = ImageTk.PhotoImage(self.image.resize((400, 400), Image.ANTIALIAS))
            # self.image_tk = ImageTk.PhotoImage(self.image.resize((400, 400), Image.Resampling.LANCZOS))

            self.canvas_original.create_image(200, 200, image=self.image_tk)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {e}")

    def zoom_image(self):
        if not self.image:
            messagebox.showwarning("Warning", "Please upload an image first!")
            return

        coords = self.coord_entry.get().split(",")
        if len(coords) != 2:
            messagebox.showwarning("Warning", "Please enter valid coordinates (x, y)!")
            return

        try:
            x, y = int(coords[0]), int(coords[1])
            width, height = self.image.size

            # Define zoom area
            left = max(0, x)
            top = max(0, y)
            right = min(width, x + 100)
            bottom = min(height, y + 100)

            zoom_area = self.image.crop((left, top, right, bottom))

            # Resize for visualization
            self.zoomed_image = zoom_area.resize((400, 400), Image.NEAREST)
            self.zoomed_image_tk = ImageTk.PhotoImage(self.zoomed_image)

            # Display zoomed image
            self.canvas_zoom.create_image(200, 200, image=self.zoomed_image_tk)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to zoom image: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = PixelPickerZoomApp(root)
    root.mainloop()
