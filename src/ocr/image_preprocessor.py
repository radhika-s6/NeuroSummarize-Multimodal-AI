import cv2
import numpy as np

def preprocess_image(self, image_path):
    """Preprocess image for better OCR results"""
    img = cv2.imread(str(image_path))
        
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Noise removal
    denoised = cv2.medianBlur(gray, 5)
    
    # Thresholding
    thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    
    # Morphological operations
    kernel = np.ones((1, 1), np.uint8)
    processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    return processed