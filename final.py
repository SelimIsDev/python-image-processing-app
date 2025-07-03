import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
from PIL import Image, ImageTk
import os                                     
import glob, shutil
import cv2
import math
import numpy as np
import random

class ImageApp:
    # Pics klasörünü sil
    shutil.rmtree("Pics", ignore_errors=True)

    # Pics klasörünü tekrar oluştur
    os.makedirs("Pics", exist_ok=True)

    def __init__(self, master):
        # Title
        self.master = master
        self.master.title("Resim Yükleme Uygulaması")
        master.geometry("1920x1080")

        # Create Paint Area
        self.canvas = tk.Canvas(master, width=1080, height=480, bg="gray")
        self.canvas.pack()
        
        self.firstBox = tk.LabelFrame(master, text="")
        self.firstBox.pack(side=tk.TOP, fill=tk.X, anchor=tk.N, padx=(15,15), pady=15)

        self.secondBox = tk.LabelFrame(master, text="")
        self.secondBox.pack(side=tk.TOP, fill=tk.X, anchor=tk.N, padx=(15,15), pady=15)
        
####### 1 (MAIN OPERATIONS)
        self.group_box1 = tk.LabelFrame(self.firstBox, text="Ana İşlemler")
        self.group_box1.pack(side=tk.LEFT, anchor=tk.N, padx=(15,5), pady=15)
        
        # Upload düğmesini oluştur ve yerleştir
        self.load_button = tk.Button(self.group_box1, text="Resim Yükle", command=self.load_image, width=12, height=2)
        self.load_button.pack(side=tk.TOP, padx=5, pady=5)

        
        self.undo_button = tk.Button(self.group_box1, text="Geri Al", command=self.returnDelete_picture, compound=tk.LEFT, width=12, height=2)
        self.undo_button.pack(side=tk.TOP, padx=5, pady=5)

        self.save_button = tk.Button(self.group_box1, text="Kaydet", command=self.saveImage, compound=tk.LEFT, width=12, height=2)
        self.save_button.pack(side=tk.TOP, padx=5, pady=5)
#######
####### 2 (MORFOLOJIK)
        self.group_box2 = tk.LabelFrame(self.firstBox, text="Morfolojik İşlemler")
        self.group_box2.pack(side=tk.LEFT, anchor=tk.N, padx=(15,5), pady=15)
        #1
        self.dilate_button = tk.Button(self.group_box2, text="Genişleme", command=self.dilate, width=12, height=2)
        self.dilate_button.pack(side=tk.TOP, padx=5, pady=5)
        #2
        self.erode_button = tk.Button(self.group_box2, text="Aşınma", command=self.erode, width=12, height=2)
        self.erode_button.pack(side=tk.TOP, padx=5, pady=5)

         # yakın Button
        self.zoom_in_button = tk.Button(self.group_box2, text="Yakınlaştır", command=self.zoom_in,width=12, height=2)
        self.zoom_in_button.pack(side=tk.TOP, padx=10, pady=5)
        
        # uzak Button
        self.zoom_out_button = tk.Button(self.group_box2, text="Uzaklaştır", command=self.zoom_out,width=12, height=2)
        self.zoom_out_button.pack(side=tk.TOP,padx=10 ,pady=5)

        self.new_frame = tk.Frame(master)
        self.new_frame.pack(side=tk.TOP, padx=5, pady=10)
        # ComboBox 
        self.zoom_factor_combo = ttk.Combobox(self.group_box2, values=["1.2", "1.5", "1.75","2.0"],width=12, height=2)
        self.zoom_factor_combo.current(1)  # Default value is 1
        self.zoom_factor_combo.pack(side=tk.TOP, padx=5, pady=10)

#######
####### 4 (SIMPLE OPERATIONS)
        self.group_box4 = tk.LabelFrame(self.firstBox, text="Temel İşlemler")
        self.group_box4.pack(side=tk.LEFT, anchor=tk.N, padx=(15,5), pady=15)

        # Binary Convertion
        self.convert_button = tk.Button(self.group_box4, text="Binary\nDönüşüm", command=self.binary_convert, width=12, height=2)
        self.convert_button.pack(side=tk.TOP, padx=5, pady=5)

        # Gray Conversion
        self.convert_button = tk.Button(self.group_box4, text="Gri Dönüşüm", command=self.GrayConversion, width=12, height=2)
        self.convert_button.pack(side=tk.TOP, padx=5, pady=10)

        # -
        

#######
####### 5 (Aritmetic Operations)
        self.group_box5 = tk.LabelFrame(self.firstBox, text="Aritmetik İşlemler")
        self.group_box5.pack(side=tk.LEFT, anchor=tk.N, padx=(15,5), pady=15)
        
        self.convert_button = tk.Button(self.group_box5, text="Adding", command=self.Adding, width=12, height=2)
        self.convert_button.pack(side=tk.TOP, padx=5, pady=10)

        self.convert_button = tk.Button(self.group_box5, text="Subtract", command=self.Subtract, width=12, height=2)
        self.convert_button.pack(side=tk.TOP, padx=5, pady=10)

        self.convert_button = tk.Button(self.group_box5, text="Multiply", command=self.Multiply, width=12, height=2)
        self.convert_button.pack(side=tk.TOP, padx=5, pady=10)

        self.convert_button = tk.Button(self.group_box5, text="Divide", command=self.Divide, width=12, height=2)
        self.convert_button.pack(side=tk.TOP, padx=5, pady=10)

#######
####### 7 (FILTER)
        self.group_box7 = tk.LabelFrame(self.firstBox, text="Renk Değişimleri")
        self.group_box7.pack(side=tk.LEFT, anchor=tk.N, padx=(15,5), pady=15)

        ## Mean Filter
        self.mean_filter_button = tk.Button(self.group_box7, text="Mean Filtre", command=self.mean_filter, width=12, height=2)
        self.mean_filter_button.pack(side=tk.TOP, padx=5, pady=5)
        
        ## Median Filter
        self.median_filter_button = tk.Button(self.group_box7, text="Medyan Filtre", command=self.median_filter, width=12, height=2)
        self.median_filter_button.pack(side=tk.TOP, padx=5, pady=5)

        ## KONTRAST ARTIRMA
        self.contrast_button = tk.Button(self.group_box7, text="Kontrastı Artır", command=self.adjust_contrast, width=12, height=2)
        self.contrast_button.pack(side=tk.TOP, padx=5, pady=5)

        ## HSV Dönüşümü        
        self.hsv_button = tk.Button(self.group_box7, text="HSV Dönüştür", command=lambda: self.hsv_dönüsüm(self.resized_image), width=12, height=2)
        self.hsv_button.pack(pady=10)

        ## CMYK Dönüşümü
        self.cmyk_button = tk.Button(self.group_box7, text="CMYK Dönüştür", command=lambda: self.cmyk_dönüsüm(self.resized_image), width=12, height=2)
        self.cmyk_button.pack(pady=10)
        
      
#######
####### 6 (OTHER)
        
        self.group_box6 = tk.LabelFrame(self.firstBox, text="Diğer İşlemler")
        self.group_box6.pack(side=tk.LEFT, anchor=tk.N, padx=(15,5), pady=15)

        # Histogram Stretching
        self.stretching_button = tk.Button(self.group_box6, text="Histogram\nGerme", command=self.stretch_histogram, width=12, height=2)
        self.stretching_button.pack(side=tk.TOP, padx=5, pady=5)

        # Gürültü Ekleme 
        self.add_noise_button = tk.Button(self.group_box6, text="Gürültü Ekle", command=self.add_noise, width=12, height=2)
        self.add_noise_button.pack(side=tk.TOP, padx=5, pady=5)

        # Kenar Algılama
        self.edge_detection_button = tk.Button(self.group_box6, text="Kenar Algılama", command=self.canny_edge_detector, width=12, height=2)
        self.edge_detection_button.pack(side=tk.TOP, padx=5, pady=5)

        # Konvolüsyon
        self.edge_detection_button = tk.Button(self.group_box6, text="Konvolüsyon", command=self.apply_edge_detection, width=12, height=2)
        self.edge_detection_button.pack(side=tk.TOP, padx=5, pady=5)



####### 3 (ROTATE)
        self.group_box3 = tk.LabelFrame(self.firstBox, text="Resim Döndürme")
        self.group_box3.pack(side=tk.LEFT, anchor=tk.N, padx=(15,5), pady=15)

        ### Rotate Frame
        self.rotate_buttons_frame = tk.Frame(self.group_box3)
        self.rotate_buttons_frame.pack(side=tk.LEFT, padx=(5,15))
        
        # Rotate Rate
        self.RotateRate = tk.Entry(self.rotate_buttons_frame, width=10, font=("Helvetica", 16))
        self.RotateRate.pack(side=tk.TOP, padx=5, pady=5)
 
        # Rotate Left
        self.convert_button_left = tk.Button(self.rotate_buttons_frame, text="Sol Döndür", command=self.RotLeft, width=16, height=2)
        self.convert_button_left.pack(side=tk.TOP, padx=5, pady=10)

        # Rotate Right
        self.convert_button_right = tk.Button(self.rotate_buttons_frame, text="Sağa Döndür", command=self.RotRight, width=16, height=2)
        self.convert_button_right.pack(side=tk.TOP, padx=5, pady=10)
#######
####### 8 (MOTION BLUR)
        self.group_box8 = tk.LabelFrame(self.firstBox, text="Hareket Bulanıklığı")
        self.group_box8.pack(side=tk.LEFT, anchor=tk.N, padx=(15,5), pady=15)
        
       ### Motion Rate Frame
        self.motion_frame = tk.Frame(self.group_box8)
        self.motion_frame.pack(side=tk.LEFT, padx=(5,15))
        
        # Motion Rate
        self.MotionRate = tk.Entry(self.motion_frame, width=10, font=("Helvetica", 16))
        self.MotionRate.pack(side=tk.TOP, padx=5, pady=5)
        
        # Motion Blur
        self.convert_button = tk.Button(self.motion_frame, text="Motion Blur", command=self.motion_blur, width=16, height=2)
        self.convert_button.pack(side=tk.TOP, padx=5, pady=10)
#######
####### 9 (THRESHOLD)
        self.group_box9 = tk.LabelFrame(self.firstBox, text="Threshold")
        self.group_box9.pack(side=tk.LEFT, anchor=tk.N, padx=(15,5), pady=15)
        
        ### Threshold Frame
        self.threshold_frame = tk.Frame(self.group_box9)
        self.threshold_frame.pack(side=tk.LEFT, anchor=tk.N, padx=5)

        #Threshold Low
        self.low_threshold_entry = tk.Entry(self.threshold_frame, width=10, font=("Helvetica", 16))
        self.low_threshold_entry.pack(side=tk.TOP, padx=5, pady=5)

        #Threshold High
        self.high_threshold_entry = tk.Entry(self.threshold_frame, width=10, font=("Helvetica", 16))
        self.high_threshold_entry.pack(side=tk.TOP, padx=5, pady=5)
        
        # Thresholding
        self.convert_button = tk.Button(self.threshold_frame, text="Thresholding", command=self.dual_threshold, width=17, height=2)
        self.convert_button.pack(side=tk.LEFT, padx=5, pady=10)

        

#######
####### 10 (CROP)
        self.group_box10 = tk.LabelFrame(self.firstBox, text="Kırpma")
        self.group_box10.pack(side=tk.LEFT, anchor=tk.N, padx=(15,5), pady=15)
        
        ### Crop Frame
        self.crop_frame = tk.Frame(self.group_box10)
        self.crop_frame.pack(padx=5, pady=5)

        # Crop x1
        self.label_x1 = tk.Label(self.crop_frame, text="Sol üst X1", font=("Helvetica", 8))
        self.label_x1.pack(side=tk.TOP, padx=1, pady=1)
        self.crop_x1 = tk.Entry(self.crop_frame, width=8, font=("Helvetica", 16))
        self.crop_x1.pack(side=tk.TOP, padx=1, pady=1)

        # Crop y1
        self.label_y1 = tk.Label(self.crop_frame, text="Sol üst Y1", font=("Helvetica", 8))
        self.label_y1.pack(side=tk.TOP, padx=1, pady=1)
        self.crop_y1 = tk.Entry(self.crop_frame, width=8, font=("Helvetica", 16))
        self.crop_y1.pack(side=tk.TOP, padx=1, pady=1)

        # Crop x2
        self.label_X2 = tk.Label(self.crop_frame, text="Sağ alt X2", font=("Helvetica", 8))
        self.label_X2.pack(side=tk.TOP, padx=3, pady=3)
        self.crop_x2 = tk.Entry(self.crop_frame, width=8, font=("Helvetica", 16))
        self.crop_x2.pack(side=tk.TOP, padx=3, pady=3)

        # Crop y2

        self.label_y2 = tk.Label(self.crop_frame, text="Sağ alt Y2", font=("Helvetica", 8))
        self.label_y2.pack(side=tk.TOP, padx=3, pady=3)
        self.crop_y2 = tk.Entry(self.crop_frame, width=8, font=("Helvetica", 16))
        self.crop_y2.pack(side=tk.TOP, padx=3, pady=3)
        
        # Croping
        self.convert_button = tk.Button(self.crop_frame, text="Kırp", command=self.crop_image, width=17, height=2)
        self.convert_button.pack(side=tk.LEFT, padx=5, pady=5)
        self.zoom_factor = 1.0

        # Resmin dosya yolu ve piksel değerlerini tutacak değişkenler
        self.file_path = ""
        self.pixel_array = []


    def load_new(self):
       
        if self.file_path:
            # Seçilen dosyayı aç ve görüntüyü yükle
            image = Image.open(self.file_path)
            self.resized_image = image.resize((1080, 480)) 
            photo = ImageTk.PhotoImage(self.resized_image)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=photo)
            self.canvas.image = photo
            # Her zaman yeni bir piksel dizisi oluştur
            self.pixel_array = list(self.resized_image.getdata())
            width, height = self.resized_image.size

            
    # Normal Resim 
    def load_image(self):
        # Dosya seçme penceresini aç
        self.file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
        if self.file_path:
            # Seçilen dosyayı aç ve görüntüyü yükle
            image = Image.open(self.file_path)
            self.resized_image = image.resize((1080, 480))
            photo = ImageTk.PhotoImage(self.resized_image)

            # Resmi görüntüle
            self.canvas.create_image(0, 0, anchor=tk.NW, image=photo)
            self.canvas.image = photo

            # Resmin tüm piksel değerlerini bir diziye kaydet
            self.pixel_array = list(self.resized_image.getdata())

            self.saveOpen_picture(self.resized_image)
             
    def convolution(self, image, kernel):
        gray_image = image.convert("L")
        image_array = np.array(gray_image)
        kernel_array = np.array(kernel)

        result_array = np.zeros_like(image_array)
        m, n = kernel_array.shape
        for i in range(1, image_array.shape[0] - 1):
            for j in range(1, image_array.shape[1] - 1):
                result_array[i, j] = np.sum(image_array[i-1:i+2, j-1:j+2] * kernel_array)

        result_image = Image.fromarray(result_array)
        return result_image

    def apply_edge_detection(self):
        edge_detection_kernel = np.array([[-1, -1, -1],
                                           [-1,  8, -1],
                                           [-1, -1, -1]])
        result_image = self.convolution(self.resized_image, edge_detection_kernel)
        self.saveOpen_picture(result_image)
        self.pixel_array = []
        
    # Gürültü Ekleme
    def add_noise(self):
        if not self.resized_image:
            print("Resim yüklenmemiş.")
            return

        noisy_image = self.resized_image.copy()

        # Gürültü ekleme işlemi
        pixels = noisy_image.load()
        width, height = noisy_image.size
        for x in range(width):
            for y in range(height):
                rand = random.random()
                if rand < 0.01:
                    pixels[x, y] = (0, 0, 0)  

        self.saveOpen_picture(noisy_image)
        self.pixel_array = []
        
  
        if not self.resized_image:
            print("Resim yüklenmemiş.")
            return
        edge_detection_kernel = np.array([[-1, -1, -1],
                                            [-1,  8, -1],
                                            [-1, -1, -1]])

        image = self.resized_image    
        gray_image= grayscale(image)
        image_array = np.array(gray_image)
        kernel_array = np.array(edge_detection_kernel)
        result_array = np.zeros_like(image_array)
        m, n = kernel_array.shape
        for i in range(1, image_array.shape[0] - 1):
            for j in range(1, image_array.shape[1] - 1):
                result_array[i, j] = np.sum(image_array[i-1:i+2, j-1:j+2] * kernel_array)
                    
        result_image = Image.fromarray(result_array)
            
        self.saveOpen_picture(result_image)
        self.pixel_array = []
   
    def gaussian_blur(self, image, kernel_size=5, sigma=1.4):
        def gaussian_kernel(size, sigma):
            kernel = np.fromfunction(
                lambda x, y: (1 / (2 * np.pi * sigma**2)) * np.exp(-((x - (size - 1) / 2)**2 + (y - (size - 1) / 2)**2) / (2 * sigma**2)),
                (size, size))
            return kernel / np.sum(kernel)
        
        kernel = gaussian_kernel(kernel_size, sigma)
        pad_size = kernel_size // 2
        image_padded = np.pad(image, ((pad_size, pad_size), (pad_size, pad_size)), mode='constant')
        blurred_image = np.zeros_like(image)

        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                blurred_image[i, j] = np.sum(kernel * image_padded[i:i+kernel_size, j:j+kernel_size])
        
        return blurred_image

    def sobel(self, image):
        Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        Ix = cv2.filter2D(image, -1, Kx)
        Iy = cv2.filter2D(image, -1, Ky)
        G = np.hypot(Ix, Iy)
        G = G / G.max() * 255
        theta = np.arctan2(Iy, Ix)
        return (G, theta)

    def non_max_suppression(self, image, D):
        M, N = image.shape
        Z = np.zeros((M, N), dtype=np.int32)
        angle = D * 180. / np.pi
        angle[angle < 0] += 180

        for i in range(1, M - 1):
            for j in range(1, N - 1):
                q = 255
                r = 255
                if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                    q = image[i, j + 1]
                    r = image[i, j - 1]
                elif (22.5 <= angle[i, j] < 67.5):
                    q = image[i + 1, j - 1]
                    r = image[i - 1, j + 1]
                elif (67.5 <= angle[i, j] < 112.5):
                    q = image[i + 1, j]
                    r = image[i - 1, j]
                elif (112.5 <= angle[i, j] < 157.5):
                    q = image[i - 1, j - 1]
                    r = image[i + 1, j + 1]

                if (image[i, j] >= q) and (image[i, j] >= r):
                    Z[i, j] = image[i, j]
                else:
                    Z[i, j] = 0

        return Z

    def threshold(self, image, lowThreshold, highThreshold):
        M, N = image.shape
        res = np.zeros((M, N), dtype=np.int32)

        strong = 255
        weak = 25

        strong_i, strong_j = np.where(image >= highThreshold)
        zeros_i, zeros_j = np.where(image < lowThreshold)

        weak_i, weak_j = np.where((image <= highThreshold) & (image >= lowThreshold))

        res[strong_i, strong_j] = strong
        res[weak_i, weak_j] = weak

        return (res, weak, strong)

    def hysteresis(self, image, weak, strong=255):
        M, N = image.shape
        for i in range(1, M - 1):
            for j in range(1, N - 1):
                if (image[i, j] == weak):
                    if ((image[i + 1, j - 1] == strong) or (image[i + 1, j] == strong) or (image[i + 1, j + 1] == strong)
                            or (image[i, j - 1] == strong) or (image[i, j + 1] == strong)
                            or (image[i - 1, j - 1] == strong) or (image[i - 1, j] == strong) or (image[i - 1, j + 1] == strong)):
                        image[i, j] = strong
                    else:
                        image[i, j] = 0
        return image

    def canny_edge_detector(self):
        if not self.resized_image:
            print("Resim yüklenmemiş.")
            return
        low_threshold = 22
        high_threshold = 32
        image = self.resized_image
        gray_image = np.array(image.convert('L'))
        
        blurred_image = self.gaussian_blur(gray_image)
        gradient_magnitude, gradient_direction = self.sobel(blurred_image)
        non_max_img = self.non_max_suppression(gradient_magnitude, gradient_direction)
        thresh_img, weak, strong = self.threshold(non_max_img, low_threshold, high_threshold)
        img_final = self.hysteresis(thresh_img, weak, strong)
        
        img_pil = Image.fromarray(img_final.astype(np.uint8))
        self.saveOpen_picture(img_pil)
        self.pixel_array = []



    # KIRPMA        
    def crop_image(self):
        
           # x1, y1, x2, ve y2 giriş kutularından değerleri al
            x1 = int(self.crop_x1.get())
            y1 = int(self.crop_y1.get())
            x2 = int(self.crop_x2.get())
            y2 = int(self.crop_y2.get())
            
            # Resmi kırp
            cropped_image = self.manual_crop(self.resized_image, x1, y1, x2, y2)

            photo = ImageTk.PhotoImage(cropped_image)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=photo)
            self.canvas.image = photo
    
    def manual_crop(self, image, x1, y1, x2, y2):
        width, height = x2 - x1, y2 - y1
        cropped_image = Image.new("RGB", (width, height))

        for x in range(width):
            for y in range(height):
                pixel = image.getpixel((x1 + x, y1 + y))
                cropped_image.putpixel((x, y), pixel)
                    
        return cropped_image
    
    ### RENK UZAYI DÖNÜŞÜMÜ

    def rgb_hsv(self, rgb):
        r,g,b=rgb
        r, g, b = r/255.0, g/255.0, b/255.0
        mx = max(r, g, b)
        mn = min(r, g, b)
        df = mx-mn
        if mx == mn:
            h = 0
        elif mx == r:
            h = (60 * ((g-b)/df) + 360) % 360
        elif mx == g:
            h = (60 * ((b-r)/df) + 120) % 360
        elif mx == b:
            h = (60 * ((r-g)/df) + 240) % 360
        if mx == 0:
            s = 0
        else:
            s = (df/mx)*100
        v = mx*100
        return h, s, v

    def rgb_cmyk(self, rgb):
        
        r, g, b = rgb
        r_norm, g_norm, b_norm = r / 255.0, g / 255.0, b / 255.0

        k = 1 - max(r_norm, g_norm, b_norm)
        if k == 1:
            c = 0
            m = 0
            y = 0
        else:
            c = (1 - r_norm - k) / (1 - k)
            m = (1 - g_norm - k) / (1 - k)
            y = (1 - b_norm - k) / (1 - k)

        return c, m, y, k

    def cmyk_dönüsüm(self, image):
        cmyk_image = Image.new("CMYK", image.size)
        for x in range(image.width):
            for y in range(image.height):
                r, g, b = image.getpixel((x, y))
                cyan, magenta, yellow, black = self.rgb_cmyk((r, g, b))
                cmyk_image.putpixel((x, y), (int(cyan * 255), int(magenta * 255), int(yellow * 255), int(black * 255)))
        cmyk_image.show()

    def hsv_dönüsüm(self, image):
        hsv_image = Image.new("HSV", image.size)
        for x in range(image.width):
            for y in range(image.height):
                r, g, b = image.getpixel((x, y))
                h, s, v = self.rgb_hsv((r, g, b))
                hsv_image.putpixel((x, y), (int(h), int(s * 255), int(v * 255)))
        hsv_image.show()

    
##YAKINLAŞTIRMA UZAKLAŞTIRMA
   

    def display_image(self, image):
        photo = ImageTk.PhotoImage(image)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=photo)
        self.canvas.image = photo
        
    def yakin_uzaklastir_fonk(self, image, factor):
        original_width, original_height = image.size
        new_width = int(original_width * factor)
        new_height = int(original_height * factor)
        
        zoomed_image = Image.new("RGB", (new_width, new_height))
        
        for y in range(new_height):
            for x in range(new_width):
                original_x = int(x / factor)
                original_y = int(y / factor)
                if original_x >= original_width:
                    original_x = original_width - 1
                if original_y >= original_height:
                    original_y = original_height - 1
                pixel = image.getpixel((original_x, original_y))
                zoomed_image.putpixel((x, y), pixel)
        
        return zoomed_image

    def zoom_in(self):
        if self.resized_image:
            zoom_factor_text = float(self.zoom_factor_combo.get())
            self.zoom_factor *= zoom_factor_text
            zoomed_image = self.yakin_uzaklastir_fonk(self.resized_image, self.zoom_factor)
            self.display_image(zoomed_image)

    def zoom_out(self):
        if self.resized_image:
            zoom_factor_text = float(self.zoom_factor_combo.get())
            self.zoom_factor /= zoom_factor_text
            zoomed_image = self.yakin_uzaklastir_fonk(self.resized_image, self.zoom_factor)
            self.display_image(zoomed_image)

    def adjust_contrast(self):
        if self.resized_image:
            adjusted_image = self.adjust_contrast_custom(self.resized_image, 1.5)
            self.saveOpen_picture(adjusted_image)
            self.pixel_array = []
        else:
            messagebox.showerror("Hata", "Lütfen önce bir resim yükleyin.")

    def adjust_contrast_custom(self, image, factor):
         width, height = image.size
         adjusted_pixels = []
         for y in range(height):
            for x in range(width):
                r, g, b = image.getpixel((x, y))
                adjusted_r = int(r * factor)
                adjusted_g = int(g * factor)
                adjusted_b = int(b * factor)
                adjusted_pixels.append((adjusted_r, adjusted_g, adjusted_b))
         adjusted_image = Image.new("RGB", (width, height))
         adjusted_image.putdata(adjusted_pixels)
         return adjusted_image
        
    def mean_filter(self):
        if not self.resized_image:
            print("Resim yüklenmemiş.")
            return

        image_array = np.array(self.resized_image.convert("L"))
        kernel_size = 3
        pad_width = kernel_size // 2
        padded_image = np.pad(image_array, pad_width, mode='constant', constant_values=0)
        filtered_image = np.zeros_like(image_array)

        for y in range(image_array.shape[0]):
            for x in range(image_array.shape[1]):
                kernel = padded_image[y:y+kernel_size, x:x+kernel_size]
                filtered_image[y, x] = np.mean(kernel)

        mean_filtered_image = Image.fromarray(filtered_image)
        self.saveOpen_picture(mean_filtered_image)
        self.pixel_array = []
        
    def median_filter(self):
        if not self.resized_image:
            print("Resim yüklenmemiş.")
            return

        image_array = np.array(self.resized_image.convert("L"))
        kernel_size = 3
        pad_width = kernel_size // 2
        padded_image = np.pad(image_array, pad_width, mode='constant', constant_values=0)
        filtered_image = np.zeros_like(image_array)

        for y in range(image_array.shape[0]):
            for x in range(image_array.shape[1]):
                kernel = padded_image[y:y+kernel_size, x:x+kernel_size]
                filtered_image[y, x] = np.median(kernel)

        median_filtered_image = Image.fromarray(filtered_image)
        self.saveOpen_picture(median_filtered_image)
        self.pixel_array = []

    # BINARY DONUSUM
    def binary_convert(self):
        if not self.resized_image:
            print("Resim yüklenmemiş.")
            return

        pixels = list(self.resized_image.convert("L").getdata())

        threshold = 128
        binary_pixels = [0 if pixel < threshold else 255 for pixel in pixels]
        binary_image = Image.new("L", (1080, 480))
        binary_image.putdata(binary_pixels)

        self.saveOpen_picture(binary_image)
        self.pixel_array = []

    def stretch_histogram(self):
        if not self.resized_image:
            print("Resim yüklenmemiş.")
            return

        pixels = list(self.resized_image.convert("L").getdata())

        min_pixel = min(pixels)
        max_pixel = max(pixels)
        
        stretched_pixels = [(pixel - min_pixel) * 255 / (max_pixel - min_pixel) for pixel in pixels]

        stretched_pixels = [int(pixel) for pixel in stretched_pixels]

        stretched_image = Image.new("L", (1080, 480))
        stretched_image.putdata(stretched_pixels)

        self.saveOpen_picture(stretched_image)
        self.pixel_array = []


    # MORFOLOJİK-1 GENİŞLEME
    def dilate(self):
        if not self.resized_image:
            print("Resim yüklenmemiş.")
            return

        binary_image = self.resized_image.convert("L")
        pixels = list(binary_image.getdata())
        width, height = binary_image.size

        new_pixels = [0] * len(pixels)
        for y in range(height):
            for x in range(width):
                if pixels[y * width + x] == 255:
                    for ky in range(-1, 2):
                        for kx in range(-1, 2):
                            nx, ny = x + kx, y + ky
                            if 0 <= nx < width and 0 <= ny < height:
                                new_pixels[ny * width + nx] = 255

        dilated_image = Image.new("L", (width, height))
        dilated_image.putdata(new_pixels)
        self.saveOpen_picture(dilated_image)
        self.pixel_array = []

    # MORFOLOJİK-2 AŞINMA
    def erode(self):
        if not self.resized_image:
            print("Resim yüklenmemiş.")
            return

        binary_image = self.resized_image.convert("L")
        pixels = list(binary_image.getdata())
        width, height = binary_image.size

        new_pixels = [255] * len(pixels)
        for y in range(height):
            for x in range(width):
                if pixels[y * width + x] == 0:
                    for ky in range(-1, 2):
                        for kx in range(-1, 2):
                            nx, ny = x + kx, y + ky
                            if 0 <= nx < width and 0 <= ny < height:
                                new_pixels[ny * width + nx] = 0

        eroded_image = Image.new("L", (width, height))
        eroded_image.putdata(new_pixels)
        self.saveOpen_picture(eroded_image)
        self.pixel_array = []
        
    def saveOpen_picture(self, image):
        print("Resim Kaydedildi...")
        i = 1
        while os.path.exists(f"Pics/{i}.png"):
            i += 1

        file_path = f"Pics/{i}.png"
        image.save(file_path)
        self.file_path = file_path
        self.load_new()

    def returnDelete_picture(self):
        # Pics klasöründeki tüm png dosyalarını listele
        file_list = glob.glob("Pics/*.png")
        
        # Dosya adlarından 'i' değerlerini çıkar
        indices = [int(file.split("\\")[-1].split(".")[0]) for file in file_list]
        
        # En büyük 'i' değerine sahip dosyayı bul
        max_index = max(indices)
        file_delete = f"Pics/{max_index}.png"
        repNum = max_index - 1
        file_replace = f"Pics/{repNum}.png"
        self.file_path = file_replace
        self.load_new()

        pics_files = os.listdir("Pics/")
        
        os.remove(file_delete)

        print(len(pics_files))
        if len(pics_files) <= 1:
            self.undo_button.config(state=tk.DISABLED)

    def saveImage(self):
        i = 1
        while os.path.exists(f"Pics/{i}.png"):
            i += 1

        try:
            os.mkdir("Saved")
            print("Kaydedilecek Dosya Ayarlandı...")
        except:
            print("Dosya Zaten Mevcut!")
        
        j = 1
        while True:
            if not os.path.exists(f"Saved/image{j}.png"):
                try:
                    image = Image.open(f"Pics/{i-1}.png")
                    image.save(f"Saved/image{j}.png")
                    print("Resim Kaydedildi...")
                    break
                except FileNotFoundError:
                    print("Resim Zaten Kaydedildi!")
                    break
                    
            j += 1


    # THRESH_HOLDİNG
    def dual_threshold(self):
        if not self.resized_image:
            print("Resim yüklenmemiş.")
            return

        low_threshold = self.low_threshold_entry.get()
        high_threshold = self.high_threshold_entry.get()

        if not low_threshold and not high_threshold:
            print("Threshold Değerleri Girilmemiş!!!")
        else:
            low_threshold = int(low_threshold)
            high_threshold = int(high_threshold)

            self.GrayConversion()

            image = self.resized_image
            pixels = image.load()

            width, height = image.size

            for x in range(width):
                for y in range(height):
                    r, g, b = pixels[x, y]
                    pixel_value = int(0.2989 * r + 0.5870 * g + 0.1140 * b)
                    

                    if pixel_value < low_threshold:
                        pixels[x, y] = (0, 0, 0)
                    elif pixel_value > high_threshold:
                        pixels[x, y] = (255, 255, 255)
                    else:
                        pixels[x, y] = (0, 0, 0)

            self.saveOpen_picture(image)
            self.pixel_array = []

    def motion_blur(self):
        if not self.resized_image:
            print("Resim yüklenmemiş.")
            return

        kernel_size = int(self.MotionRate.get())
        image = self.resized_image
        width, height = image.size

        kernel = np.zeros((kernel_size, kernel_size))
        kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
        kernel = kernel / kernel_size

        pad = kernel_size // 2
        padded_image = np.pad(np.array(image), ((pad, pad), (pad, pad), (0, 0)), mode='constant', constant_values=0)

        blurred_image = np.zeros_like(np.array(image))

        for i in range(height):
            for j in range(width):
                for k in range(3):
                    blurred_image[i, j, k] = np.sum(kernel * padded_image[i:i + kernel_size, j:j + kernel_size, k])

        self.saveOpen_picture(Image.fromarray(np.uint8(blurred_image)))
        self.pixel_array = []

    def arithmetical_operations(self, operation):
        if not self.resized_image:
            print("Birinci resim yüklenmemiş.")
            return

        image_path2 = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.gif")])
        if not image_path2:
            print("İkinci resim seçilmedi.")
            return

        image1 = self.resized_image.convert("RGB")
        image2 = Image.open(image_path2).convert("RGB")

        if image1.size != image2.size:
            image1 = image1.resize(image2.size)
        result_image = Image.new("RGB", image1.size)

        for y in range(image1.height):
            for x in range(image1.width):
                pixel1 = image1.getpixel((x, y))
                pixel2 = image2.getpixel((x, y))
                r1, g1, b1 = pixel1
                r2, g2, b2 = pixel2
                
                if operation == "add":
                    result_pixel = (min(r1 + r2, 255), min(g1 + g2, 255), min(b1 + b2, 255))

                elif operation == "subtract":
                    result_pixel = (max(r1 - r2, 0), max(g1 - g2, 0), max(b1 - b2, 0))

                elif operation == "multiply":
                    result_pixel = (int(r1 * r2 / 255), int(g1 * r2 / 255), int(b1 * r2 / 255))

                elif operation == "divide":
                    if r2 == 0:
                        r2 = 1
                    if g2 == 0:
                        g2 = 1
                    if b2 == 0:
                        b2 = 1
                    result_pixel = (int(r1 / r2 * 255), int(g1 / r2 * 255), int(b1 / r2 * 255))

                else:
                    raise ValueError("Invalid operation")

                result_image.putpixel((x, y), result_pixel)

        self.saveOpen_picture(result_image)
        self.pixel_array = []

    def Adding(self):
        self.arithmetical_operations("add")

    def Subtract(self):
        self.arithmetical_operations("subtract")

    def Multiply(self):
        self.arithmetical_operations("multiply")

    def Divide(self):
        self.arithmetical_operations("divide")

    # Gri Dönüşüm
    def GrayConversion(self):
        if not self.resized_image:
            print("Resim yüklenmemiş.")
            return
        
        if self.resized_image.mode == 'L':
            print("Resim zaten gri tonlamalı.")
            self.resized_image = self.resized_image.convert('RGB')
            
        width, height = self.resized_image.size

        for y in range(height):
            for x in range(width):
                r, g, b = self.resized_image.getpixel((x, y))

                gray_value = int(0.2989 * r + 0.5870 * g + 0.1140 * b)

                self.resized_image.putpixel((x, y), (gray_value, gray_value, gray_value))

        photo = ImageTk.PhotoImage(self.resized_image)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=photo)
        self.canvas.image = photo
        
        self.saveOpen_picture(self.resized_image)
        self.pixel_array = []

    # Döndürme
    def Rotate(self, direction):

        if not self.resized_image:
            print("Resim yüklenmemiş.")
            return

        rotImage = self.resized_image
        
        width, height = rotImage.size

        rotated_img = Image.new('RGB', (width, height))
        angle = float(self.RotateRate.get())
        print(angle)
        if (direction == "left"):
            angle = -1*angle

        for x in range(width):
            for y in range(height):
                new_x = int((x - width / 2) * math.cos(math.radians(angle)) - (y - height / 2) * math.sin(math.radians(angle)) + width / 2)
                new_y = int((x - width / 2) * math.sin(math.radians(angle)) + (y - height / 2) * math.cos(math.radians(angle)) + height / 2)

                if 0 <= new_x < width and 0 <= new_y < height:
                    rotated_img.putpixel((x, y), self.resized_image.getpixel((new_x, new_y)))
                    
        self.saveOpen_picture(rotated_img)
        self.pixel_array = []

    def RotLeft(self):
        self.Rotate("left")
        
    def RotRight(self):
        self.Rotate("right")
            


# Ana uygulamayı oluştur
root = tk.Tk()
app = ImageApp(root)
root.mainloop()
