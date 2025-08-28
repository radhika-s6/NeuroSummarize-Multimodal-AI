import os
import re
import cv2
import pytesseract
import easyocr
from pathlib import Path
from typing import List, Tuple
from PIL import Image, ImageFilter, ImageOps
import fitz  # PyMuPDF

def preprocess_image(img_path: str) -> Image.Image:
    img = Image.open(img_path).convert("L")
    img = img.filter(ImageFilter.MedianFilter(size=3))
    return img

def read_with_tesseract(pil_img: Image.Image) -> str:
    config = r'--oem 3 --psm 6'
    return pytesseract.image_to_string(pil_img, lang='eng', config=config)

def read_with_easyocr(img_path: str) -> str:
    reader = easyocr.Reader(['en'], gpu=False)
    result = reader.readtext(img_path, detail=0, paragraph=True)
    return "\n".join(result)

def ocr_image(img_path: str, engine: str = "tesseract") -> str:
    if engine == "easyocr":
        return read_with_easyocr(img_path)
    pil_img = preprocess_image(img_path)
    return read_with_tesseract(pil_img)

def save_text(text: str, out_path: str):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True
    )
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)

def pdf_to_images(pdf_path: str, out_dir: str, dpi: int = 200) -> List[str]:
    """
    Convert PDF pages to PNG images using PyMuPDF (no Poppler needed).
    """
    pdf_path = Path(pdf_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(pdf_path)
    out_paths = []
    for i, page in enumerate(doc):
        pix = page.get_pixmap(dpi=dpi)
        out_file = out_dir / f"{pdf_path.stem}_p{i+1:03d}.png"
        pix.save(str(out_file))
        out_paths.append(str(out_file))

    return out_paths

def ocr_folder(in_dir: str, out_dir: str, engine: str = "tesseract") -> List[Tuple[str, str]]:
    results = []
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    for name in os.listdir(in_dir):
        p = os.path.join(in_dir, name)
        if not os.path.isfile(p):
            continue
        ext = os.path.splitext(name)[1].lower()
        if ext in [".png", ".jpg", ".jpeg", ".tiff", ".bmp"]:
            text = ocr_image(p, engine=engine)
            out_txt = os.path.join(out_dir, os.path.splitext(name)[0] + ".txt")
            save_text(text, out_txt)
            results.append((p, out_txt))
        elif ext == ".pdf":
            # Convert each page then OCR
            img_dir = os.path.join(out_dir, os.path.splitext(name)[0] + "_pages")
            img_paths = pdf_to_images(p, img_dir)
            combined = []
            for ip in img_paths:
                combined.append(ocr_image(ip, engine=engine))
            out_txt = os.path.join(out_dir, os.path.splitext(name)[0] + ".txt")
            save_text("\n\n".join(combined), out_txt)
            results.append((p, out_txt))
        else:
            print(f"Skip unsupported: {name}")
    return results
