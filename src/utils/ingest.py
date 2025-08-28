# src/utils/ingest.py
import os, io, tempfile, gzip
from pathlib import Path
from typing import Tuple
import pandas as pd

from docx import Document  # python-docx
from src.text_utils import basic_clean, clip_for_model
from src.ocr import ocr_image, pdf_to_images  # existing helpers

SUPPORTED_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tiff", ".bmp"}
SUPPORTED_TEXT_EXTS  = {".txt"}
SUPPORTED_DOCX_EXTS  = {".docx"}
SUPPORTED_TABLE_EXTS = {".csv", ".tsv", ".tsv.gz", ".csv.gz", ".xlsx"}
SUPPORTED_PDF_EXTS   = {".pdf"}

def sanitize_filename(name: str) -> str:
    safe = "".join(c if c.isalnum() or c in ("-", "_", ".") else "_" for c in name)
    return safe[:120] or "file"

def detect_file_type(filename: str) -> str:
    fn = filename.lower()
    for ext in SUPPORTED_IMAGE_EXTS:
        if fn.endswith(ext): return "image"
    if fn.endswith(".pdf"): return "pdf"
    if fn.endswith(".docx"): return "docx"
    if fn.endswith(".txt"): return "txt"
    if any(fn.endswith(ext) for ext in [".csv",".tsv",".csv.gz",".tsv.gz",".xlsx"]): return "table"
    return "unknown"

def _read_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def _read_docx(path: str) -> str:
    doc = Document(path)
    return "\n".join(p.text for p in doc.paragraphs)

def _read_table_like(path: str) -> pd.DataFrame:
    fn = path.lower()
    if fn.endswith(".xlsx"):
        return pd.read_excel(path)
    if fn.endswith(".csv") or fn.endswith(".csv.gz"):
        return pd.read_csv(path, sep=",", compression="infer")
    if fn.endswith(".tsv") or fn.endswith(".tsv.gz"):
        return pd.read_csv(path, sep="\t", compression="infer")
    # fallback try:
    return pd.read_csv(path, sep=None, engine="python")

def _table_to_report(df: pd.DataFrame, cap_rows: int = 25) -> str:
    """
    Turn participant/clinical tables into pseudo-reports.
    Heuristics align with your OpenNeuro/TCIA usage.  :contentReference[oaicite:3]{index=3}
    """
    cols = {c.lower(): c for c in df.columns}
    id_col  = cols.get("participant_id") or cols.get("subject_id") or list(df.columns)[0]
    dx_col  = cols.get("diagnosis") or cols.get("group")
    age_col = cols.get("age")
    sex_col = cols.get("sex")

    lines = []
    for i, row in df.head(cap_rows).iterrows():
        sid = str(row.get(id_col, f"row-{i}"))
        dx  = str(row.get(dx_col, "unknown")) if dx_col else "unknown"
        age = str(row.get(age_col, "NA")) if age_col else "NA"
        sex = str(row.get(sex_col, "NA")) if sex_col else "NA"
        lines.append(
            f"Patient: {sid}\n"
            f"Modality: MRI (from dataset metadata)\n"
            f"Diagnosis/Group: {dx}\n"
            f"Age: {age}; Sex: {sex}\n"
            f"Findings: (derived from dataset / metadata)\n"
            f"Recommendation: Correlate with clinical phenotype.\n"
            "----"
        )
    if not lines:
        # Generic summary if unknown columns
        lines = [df.head(cap_rows).to_csv(index=False)]
    return "\n".join(lines)

def extract_text_from_file(path: str, engine: str = "tesseract") -> Tuple[str, str]:
    """
    Returns (text, mode) where mode is one of {'ocr', 'pdf+ocr', 'text', 'docx', 'table'}.
    Applies basic_clean + clip_for_model to keep models safe.  :contentReference[oaicite:4]{index=4}
    """
    kind = detect_file_type(path)
    if kind == "image":
        text = ocr_image(path, engine=engine)
        return clip_for_model(basic_clean(text)), "ocr"

    if kind == "pdf":
        # Multi-page safe: convert to images then OCR per page
        tmp_img_dir = os.path.join(tempfile.mkdtemp(), "pdf_pages")
        page_imgs = pdf_to_images(path, tmp_img_dir)
        pages_text = []
        for i, ip in enumerate(page_imgs, start=1):
            t = ocr_image(ip, engine=engine)
            pages_text.append(f"=== Page {i} ===\n{t}")
        return clip_for_model(basic_clean("\n\n".join(pages_text))), "pdf+ocr"

    if kind == "docx":
        return clip_for_model(basic_clean(_read_docx(path))), "docx"

    if kind == "txt":
        return clip_for_model(basic_clean(_read_txt(path))), "text"

    if kind == "table":
        df = _read_table_like(path)
        return clip_for_model(basic_clean(_table_to_report(df))), "table"

    raise ValueError(f"Unsupported file type for: {path}")
