# 🖼️ Custom Image Processing Application – Python GUI (No External Libraries)

![App Screenshot](https://github.com/user-attachments/assets/c3368270-2435-4134-bf66-2992f2af2fa1)

This is a Python-based **image processing application** that features a graphical user interface and provides a wide range of common image operations **without using any prebuilt image processing libraries** like OpenCV or PIL. All functions are manually implemented, making it ideal for educational or low-level algorithm demonstration purposes.

---

## 🎯 Key Features

The application is divided into functional sections, each performing a specific group of image processing tasks:

### 📁 Main Functions
- **Resim Yükle (Load Image)**: Load an image from your local system.  
- **Geri Al (Undo)**: Revert the last operation.  
- **Kaydet (Save)**: Save the edited image.

### 🧱 Morphological Operations
- **Genişleme (Dilation)**  
- **Aşınma (Erosion)**  
- **Yakınlaştır / Uzaklaştır (Zoom In / Out)** — with adjustable zoom level.

### ⚙️ Basic Operations
- **Binary Dönüşüm (Binary Conversion)**  
- **Gri Dönüşüm (Grayscale Conversion)**

### ➕ Arithmetic Operations
- **Add / Subtract / Multiply / Divide** — pixel-wise image math.

### 🎨 Color Transformations
- **Mean Filter / Median Filter**  
- **Contrast Adjustment**  
- **HSV & CMYK Conversion**

### 🧪 Other Filters
- **Histogram Stretching**  
- **Noise Addition**  
- **Edge Detection (e.g., Sobel)**  
- **Convolution** (with custom kernel)

### 🔁 Image Rotation
- **Rotate Left / Right**

### 📸 Motion Blur
- Custom implementation of motion blur.

### 🎚 Thresholding
- Apply binary threshold with user-defined limit.

### ✂️ Cropping
- Manually define crop coordinates using (X1, Y1) to (X2, Y2).

---

## 🧑‍💻 How It Works

- Built with **Tkinter** for GUI.
- All image operations are manually performed on pixel arrays using pure Python (`list[list[tuple]]`, etc.).
- Image is updated instantly after each operation.
- No dependencies on external libraries like `OpenCV`, `NumPy`, or `PIL`.

---

## ⚠️ Educational Purpose

This project is ideal for learning:
- How image filters work at the pixel level  
- Manual convolution  
- Color model conversions (RGB to HSV, CMYK)  
- Morphological transformations like dilation and erosion  
- Implementing your own zoom and rotate logic

---

## 🛠 Technologies Used

- **Python 3.x**  
- **Tkinter** for GUI  
- Pure Python logic for all algorithms

---

## 📂 Future Improvements

- Real-time filter preview before applying  
- Better performance for high-res images  
- Mask-based selective editing  
- Export filter logs for reproducibility

---

## 
