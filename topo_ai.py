"""
Загрузка топографических карт (PDF/PNG/JPG) и OCR высот.
"""

import cv2
import numpy as np
import fitz  # PyMuPDF
from pathlib import Path


def load_image(path, dpi=300):
    """Загружает PDF или изображение, возвращает BGR-картинку."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Файл не найден: {path}")

    if p.suffix.lower() == ".pdf":
        print(f"Конвертация PDF → изображение (PyMuPDF, {dpi} DPI)...")
        doc = fitz.open(str(p))
        if len(doc) == 0:
            raise ValueError("PDF пустой")
        page = doc.load_page(0)
        pix = page.get_pixmap(dpi=dpi)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, pix.n
        )
        if pix.n == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        elif pix.n == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        doc.close()
    else:
        img = cv2.imread(str(p))

    if img is None:
        raise ValueError(f"Не удалось открыть: {path}")
    return img


def extract_heights(image):
    """Ищет на карте числовые подписи (высоты) через OCR."""
    import easyocr
    print("Инициализация OCR (первый запуск скачает модель ~100 МБ)...")
    reader = easyocr.Reader(["ru", "en"], gpu=False)
    print("Распознавание текста (1–3 минуты)...")
    results = reader.readtext(image)

    heights = []
    for bbox, text, prob in results:
        cleaned = text.replace(",", ".").replace(" ", "").strip()
        try:
            value = float(cleaned)
        except ValueError:
            continue
        if 0 < value < 9000 and len(cleaned) >= 2:
            cx = int((bbox[0][0] + bbox[2][0]) / 2)
            cy = int((bbox[0][1] + bbox[2][1]) / 2)
            heights.append({
                "value": value,
                "x_px": cx,
                "y_px": cy,
                "confidence": round(float(prob), 3),
            })
    return heights