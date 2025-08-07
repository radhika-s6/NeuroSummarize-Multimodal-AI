import cv2
import numpy as np
import re
from PIL import Image, ImageFilter

def denoise_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)  # Force color image
    if img is None:
        raise FileNotFoundError(f"Image not found or unable to read: {image_path}")
    if img.dtype != 'uint8':
        img = img.astype('uint8')
    denoised = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    return denoised

def binarize_image(image_path):
    """
    Converts image to black-and-white using Otsu thresholding.
    """
    gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary

def correct_skew(image_path):
    """
    Deskews tilted images using contours and moments.
    """
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    coords = np.column_stack(np.where(thresh > 0))
    angle = cv2.minAreaRect(coords)[-1]

    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    (h, w) = img.shape
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    deskewed = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return deskewed

def clean_text(text):
    """
    Removes extra whitespace, artifacts, and footers from OCR text.
    """
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'\s{2,}', ' ', text)
    text = re.sub(r'\n\s*\n', '\n', text)
    return text.strip()

def remove_headers_and_footers(text, max_line_len=60):
    """
    Strips likely headers/footers based on line length heuristics.
    """
    lines = text.strip().splitlines()
    filtered = [line for line in lines if len(line.strip()) > 5 and len(line) < max_line_len]
    return "\n".join(filtered)