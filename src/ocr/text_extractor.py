import cv2
import numpy as np
from PIL import Image
import easyocr
import pytesseract
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
import torch
from src.utils import preprocessing
from src.utils.preprocessing import clean_text


class MultiModalOCR:
    def __init__(self):
        self.easyocr_reader = easyocr.Reader(['en'], verbose=False)
        self.layoutlm_processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base")
        self.layoutlm_model = LayoutLMv3ForTokenClassification.from_pretrained("microsoft/layoutlmv3-base")

    def preprocess_image(self, image_path):
        """Preprocess image for better OCR results"""
        img = cv2.imread(str(image_path))
        if img is None:
            raise FileNotFoundError(f"Image not found or unreadable: {image_path}")

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

    def extract_with_easyocr(self, image_path):
        """Extract text using EasyOCR"""
        try:
            processed_img = self.preprocess_image(image_path)
            results = self.easyocr_reader.readtext(processed_img)

            text_blocks = []
            confidence_scores = []

            for (bbox, text, confidence) in results:
                if confidence > 0.5:
                    text_blocks.append(text)
                    confidence_scores.append(confidence)

            return {
                'text': ' '.join(text_blocks),
                'confidence': np.mean(confidence_scores) if confidence_scores else 0,
                'method': 'easyocr'
            }
        except Exception as e:
            return {'text': '', 'confidence': 0, 'method': 'easyocr', 'error': str(e)}

    def extract_with_tesseract(self, image_path):
        """Extract text using Tesseract OCR"""
        try:
            processed_img = self.preprocess_image(image_path)
            text = pytesseract.image_to_string(processed_img, config='--psm 6')

            # Confidence scores
            data = pytesseract.image_to_data(processed_img, output_type=pytesseract.Output.DICT)
            confidences = [int(conf) for conf in data['conf'] if conf != '-1']
            avg_confidence = np.mean(confidences) / 100 if confidences else 0

            return {
                'text': text.strip(),
                'confidence': avg_confidence,
                'method': 'tesseract'
            }
        except Exception as e:
            return {'text': '', 'confidence': 0, 'method': 'tesseract', 'error': str(e)}

    def extract_with_layoutlm(self, image_path):
        """Extract text using LayoutLMv3"""
        try:
            image = Image.open(image_path).convert('RGB')
            encoding = self.layoutlm_processor(image, return_tensors="pt")

            with torch.no_grad():
                outputs = self.layoutlm_model(**encoding)

            tokens = self.layoutlm_processor.tokenizer.convert_ids_to_tokens(encoding["input_ids"][0])
            text = self.layoutlm_processor.tokenizer.convert_tokens_to_string(tokens)

            return {
                'text': text.strip(),
                'confidence': 0.8,
                'method': 'layoutlm'
            }
        except Exception as e:
            return {'text': '', 'confidence': 0, 'method': 'layoutlm', 'error': str(e)}

    def ensemble_extraction(self, image_path):
        """Combine results from EasyOCR, Tesseract, and LayoutLM"""
        results = {
            'easyocr': self.extract_with_easyocr(image_path),
            'tesseract': self.extract_with_tesseract(image_path),
            'layoutlm': self.extract_with_layoutlm(image_path)
        }

        best_result = max(results.values(), key=lambda x: x.get('confidence', 0) * len(x.get('text', '')))

        return {
            'final_text': best_result.get('text', ''),
            'best_method': best_result.get('method', 'none'),
            'confidence': best_result.get('confidence', 0),
            'all_results': results
        }
