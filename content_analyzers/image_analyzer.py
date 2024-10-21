from typing import Dict, List
import numpy as np
import cv2
from PIL import Image
from scipy.stats import entropy
from tf_keras.applications import VGG16
from tf_keras.applications.vgg16 import preprocess_input
from tf_keras.preprocessing import image
from .base_analyzer import BaseAnalyzer

class ImageAnalyzer(BaseAnalyzer):
    def __init__(self):
        self.model = VGG16(weights='imagenet', include_top=False)
    
    def analyze(self, img_path):
        features = {}
        
        # Load and preprocess image
        img = cv2.imread(img_path)
        pil_img = Image.open(img_path)
        
        # Basic image properties
        features.update(self._compute_basic_properties(img, pil_img))
        
        # Color analysis
        features.update(self._analyze_color(img))
        
        # Texture analysis
        features.update(self._analyze_texture(img))
        
        # Deep features
        features.update(self._extract_deep_features(img_path))
        
        return features
    
    def _compute_basic_properties(self, img, pil_img):
        return {
            "width": img.shape[1],
            "height": img.shape[0],
            "aspect_ratio": img.shape[1] / img.shape[0],
            "file_size": pil_img.size,
            "mode": pil_img.mode
        }
    
    def _analyze_color(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
        hist_flatten = hist.flatten()
        hist_norm = hist_flatten / sum(hist_flatten)
        
        return {
            "color_entropy": entropy(hist_norm),
            "dominant_color": np.unravel_index(np.argmax(hist), hist.shape)
        }
    
    def _analyze_texture(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        glcm = self._calculate_glcm(gray)
        
        contrast = np.sum(np.square(np.arange(glcm.shape[0])) * glcm)
        homogeneity = np.sum(glcm / (1 + np.square(np.arange(glcm.shape[0]))))
        energy = np.sum(np.square(glcm))
        correlation = np.sum(glcm * np.outer(np.arange(glcm.shape[0]), np.arange(glcm.shape[0])))
        
        return {
            "contrast": contrast,
            "homogeneity": homogeneity,
            "energy": energy,
            "correlation": correlation
        }
    
    def _calculate_glcm(self, img, distance=1, angle=0):
        h, w = img.shape
        glcm = np.zeros((256, 256), dtype=np.uint32)
        
        for i in range(h):
            for j in range(w):
                if i + distance < h and j + distance < w:
                    i_intensity = img[i, j]
                    j_intensity = img[i + distance, j + distance]
                    glcm[i_intensity, j_intensity] += 1
        
        return glcm / np.sum(glcm)
    
    def _extract_deep_features(self, img_path):
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        
        features = self.model.predict(x)
        return {
            "deep_features_mean": np.mean(features),
            "deep_features_std": np.std(features)
        }
