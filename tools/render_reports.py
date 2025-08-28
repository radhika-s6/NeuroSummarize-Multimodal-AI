import os
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from PIL import Image, ImageDraw, ImageFont

IN_DIR = "data/reports"
OUT_PDF = "data/reports_pdf"
OUT_IMG = "data/reports_img"

def render_pdf(txt_path: str, out_path: str):
    with open(txt_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    c = canvas.Canvas(out_path, pagesize=letter)
    width, height = letter
    y = height - 50
    for line in lines:
        c.drawString(50, y, line.strip())
        y -= 15
        if y < 50:
            c.showPage()
            y = height - 50
    c.save()

def render_png(txt_path: str, out_path: str):
    with open(txt_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Create blank white image
    img = Image.new("RGB", (1000, 1400), "white")
    draw = ImageDraw.Draw(img)

    # Use default PIL font (you can replace with a .ttf if installed)
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()

    # Wrap text
    y = 50
    for line in text.splitlines():
        draw.text((50, y), line, fill="black", font=font)
        y += 30
    img.save(out_path)

def main():
    Path(OUT_PDF).mkdir(parents=True, exist_ok=True)
    Path(OUT_IMG).mkdir(parents=True, exist_ok=True)

    for fname in os.listdir(IN_DIR):
        if not fname.endswith(".txt"):
            continue
        in_path = os.path.join(IN_DIR, fname)
        base = os.path.splitext(fname)[0]
        out_pdf = os.path.join(OUT_PDF, base + ".pdf")
        out_img = os.path.join(OUT_IMG, base + ".png")
        render_pdf(in_path, out_pdf)
        render_png(in_path, out_img)

    print(f"Rendered reports into {OUT_PDF} and {OUT_IMG}")

if __name__ == "__main__":
    main()
