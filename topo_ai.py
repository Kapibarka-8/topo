"""
Анализ топографических карт: распознавание высот, линий, измерение расстояний.

Использование:
    python topo_ai.py info --file map.pdf
    python topo_ai.py measure --file map.pdf --p1 100,200 --p2 500,600 \
        --real 50 --real-from 100,100 --real-to 400,100
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def load_image(path):
    """Загружает PDF или изображение, возвращает BGR-картинку."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Файл не найден: {path}")

    if p.suffix.lower() == ".pdf":
        from pdf2image import convert_from_path
        print("Конвертация PDF → изображение (300 DPI)...")
        pages = convert_from_path(str(p), dpi=300)
        if not pages:
            raise ValueError("PDF пустой")
        img = cv2.cvtColor(np.array(pages[0]), cv2.COLOR_RGB2BGR)
    else:
        img = cv2.imread(str(p))

    if img is None:
        raise ValueError(f"Не удалось открыть: {path}")
    return img


def add_grid(img, step=100):
    """Накладывает координатную сетку для считывания координат вручную."""
    out = img.copy()
    h, w = out.shape[:2]
    for x in range(0, w, step):
        cv2.line(out, (x, 0), (x, h), (200, 200, 200), 1)
        cv2.putText(out, str(x), (x + 2, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    for y in range(0, h, step):
        cv2.line(out, (0, y), (w, y), (200, 200, 200), 1)
        cv2.putText(out, str(y), (2, y - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    return out


def extract_heights(img):
    """Ищет на карте числовые подписи (высоты) через OCR."""
    import easyocr
    print("Инициализация OCR (первый запуск скачает модель ~100 МБ)...")
    reader = easyocr.Reader(["ru", "en"], gpu=False)
    print("Распознавание текста (1–3 минуты)...")
    results = reader.readtext(img)

    heights = []
    for bbox, text, prob in results:
        cleaned = text.replace(",", ".").replace(" ", "").strip()
        try:
            value = float(cleaned)
        except ValueError:
            continue
        # Высоты в топографии обычно в диапазоне 0–9000 м
        if 0 < value < 9000 and len(cleaned) >= 2:
            cx = int((bbox[0][0] + bbox[2][0]) / 2)
            cy = int((bbox[0][1] + bbox[2][1]) / 2)
            heights.append({
                "value": value,
                "x": cx,
                "y": cy,
                "confidence": round(float(prob), 3),
            })
    return heights


def detect_contours(img):
    """Выделяет тонкие линии (кандидаты в горизонтали)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    edges = cv2.Canny(binary, 50, 150)
    return edges


def compute_meters_per_px(p1, p2, real_distance):
    """Считает масштаб (м/пиксель) по двум точкам с известным расстоянием."""
    px = np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
    if px == 0:
        raise ValueError("Точки калибровки совпадают")
    return real_distance / px


def parse_point(s):
    """'x,y' → (int, int)."""
    x, y = s.split(",")
    return int(x.strip()), int(y.strip())


# ---------- Команды ----------

def cmd_info(args):
    """Режим info: анализ карты, сетка, высоты, линии."""
    print(f"Загрузка: {args.file}")
    img = load_image(args.file)
    print(f"Размер: {img.shape[1]}×{img.shape[0]} px")

    grid = add_grid(img, step=args.grid_step)
    grid_path = Path(args.file).stem + "_grid.png"
    cv2.imwrite(grid_path, grid)
    print(f"Сетка сохранена: {grid_path}")

    edges = detect_contours(img)
    edges_path = Path(args.file).stem + "_edges.png"
    cv2.imwrite(edges_path, edges)
    print(f"Линии сохранены: {edges_path}")

    heights = extract_heights(img)
    print(f"\nНайдено отметок высот: {len(heights)}")
    for h in heights[:30]:
        print(f"  {h['value']:>8.1f} м   @ ({h['x']}, {h['y']})   conf={h['confidence']:.2f}")
    if len(heights) > 30:
        print(f"  ... и ещё {len(heights) - 30}")

    if args.save_json:
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(heights, f, ensure_ascii=False, indent=2)
        print(f"\nJSON сохранён: {args.save_json}")

def scale_to_mpp(scale_denominator, dpi):
    """
    Переводит масштаб карты (1:N) и DPI в метры на пиксель.
    
    Формула: м/px = (N / 100) × 2.54 / DPI
    Пример: 1:2000 при 300 DPI → (2000/100) × 2.54 / 300 = 0.169 м/px
    """
    return (scale_denominator / 100.0) * 2.54 / dpi

def cmd_measure(args):
    """Режим measure: расстояние между двумя точками."""
    p1 = parse_point(args.p1)
    p2 = parse_point(args.p2)

    if args.meters_per_px:
        mpp = args.meters_per_px
        print(f"Масштаб: {mpp:.4f} м/px (задан напрямую)")
    elif args.scale and args.dpi:
        mpp = (args.scale / 100.0) * 2.54 / args.dpi
        print(f"Масштаб 1:{args.scale}, DPI={args.dpi}")
        print(f"→ Метров на пиксель: {mpp:.4f}")
    elif args.scale:
        raise ValueError("Вместе с --scale укажите --dpi (например, --dpi 300)")
    elif args.real and args.real_from and args.real_to:
        rp1 = parse_point(args.real_from)
        rp2 = parse_point(args.real_to)
        px_calib = np.sqrt((rp1[0] - rp2[0]) ** 2 + (rp1[1] - rp2[1]) ** 2)
        if px_calib == 0:
            raise ValueError("Точки калибровки совпадают")
        mpp = args.real / px_calib
        print(f"Калибровка по известному отрезку: {mpp:.4f} м/px")
    else:
        raise ValueError(
            "Укажите --scale N --dpi D, "
            "ИЛИ --meters-per-px X, "
            "ИЛИ --real X --real-from x,y --real-to x,y"
        )

    px = np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
    meters = px * mpp

    print(f"\nТочка 1: {p1}")
    print(f"Точка 2: {p2}")
    print(f"Расстояние: {px:.1f} px = {meters:.2f} м")

    if args.file:
        img = load_image(args.file)
        cv2.line(img, p1, p2, (0, 0, 255), 3)
        cv2.circle(img, p1, 8, (0, 255, 0), -1)
        cv2.circle(img, p2, 8, (255, 0, 0), -1)
        mid = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
        cv2.putText(img, f"{meters:.1f} m", mid,
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        out = Path(args.file).stem + "_measure.png"
        cv2.imwrite(out, img)
        print(f"Визуализация сохранена: {out}")


def cmd_pixels(args):
    """Считает расстояние в пикселях и метры на пиксель (для калибровки)."""
    p1 = parse_point(args.p1)
    p2 = parse_point(args.p2)
    px = float(np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2))
    print(f"Точка 1: {p1}")
    print(f"Точка 2: {p2}")
    print(f"Расстояние в пикселях: {px:.2f}")

    if args.real is not None:
        mpp = args.real / px
        print(f"\nИзвестное реальное расстояние: {args.real} м")
        print(f"→ Метров на пиксель: {mpp:.4f}")
        print(f"\nИспользуйте в measure:")
        print(f"  --meters-per-px {mpp:.4f}")

def cmd_objects(args):
    """Находит объекты на карте и выводит их центры (x, y)."""
    print(f"Загрузка: {args.file}")
    img = load_image(args.file)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    objects = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < args.min_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        cx, cy = x + w // 2, y + h // 2
        objects.append({
            "x": cx,
            "y": cy,
            "bbox": [x, y, w, h],
            "area": round(area, 1),
        })

    objects.sort(key=lambda o: -o["area"])
    print(f"\nНайдено объектов: {len(objects)}")
    print(f"{'№':>4}  {'центр (x, y)':<18}  {'площадь px²':>12}  bbox")
    for i, obj in enumerate(objects[: args.limit]):
        x, y, w, h = obj["bbox"]
        print(f"{i+1:>4}  ({obj['x']:>5}, {obj['y']:>5})       "
              f"{obj['area']:>12.1f}  [{x},{y},{w},{h}]")

    if args.save_json:
        import json
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(objects, f, ensure_ascii=False, indent=2)
        print(f"\nJSON сохранён: {args.save_json}")

    # визуализация
    vis = img.copy()
    for i, obj in enumerate(objects[: args.limit]):
        x, y, w, h = obj["bbox"]
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.circle(vis, (obj["x"], obj["y"]), 4, (0, 0, 255), -1)
        cv2.putText(vis, f"{i+1}", (x, y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    out = Path(args.file).stem + "_objects.png"
    cv2.imwrite(out, vis)
    print(f"Визуализация сохранена: {out}")


def cmd_pick(args):
    """Создаёт HTML-страницу для клика по карте и получения координат."""
    print(f"Загрузка: {args.file}")
    img = load_image(args.file)
    h, w = img.shape[:2]
    print(f"Размер: {w}×{h} px")

    # Сохраняем картинку отдельным файлом рядом с HTML
    stem = Path(args.file).stem
    img_file = f"{stem}_preview.jpg"
    ok = cv2.imwrite(img_file, img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise ValueError("Не удалось сохранить preview-картинку")
    print(f"Preview сохранён: {img_file}")

    html = f"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="utf-8">
<title>Координаты точек — {Path(args.file).name}</title>
<style>
  body {{ font-family: monospace; margin: 0; background: #111; color: #eee; }}
  #wrap {{ position: relative; }}
  #img {{ display: block; max-width: 100%; cursor: crosshair; user-select: none; }}
  #log {{ position: fixed; top: 10px; right: 10px; width: 360px;
         background: rgba(0,0,0,0.92); padding: 12px; border-radius: 8px;
         max-height: 85vh; overflow-y: auto; z-index: 10; }}
  .pt {{ color: #0f0; padding: 2px 0; }}
  button {{ margin-top: 8px; margin-right: 6px; padding: 5px 10px;
           cursor: pointer; border-radius: 4px; border: none; }}
  small {{ color: #aaa; }}
  hr {{ border-color: #333; }}
</style>
</head>
<body>
<div id="wrap">
  <img id="img" src="{img_file}" alt="map">
</div>
<div id="log">
  <b>Кликните по карте</b><br>
  <small>Размер: {w} × {h} px</small>
  <hr>
  <div id="list"></div>
  <button onclick="copyAll()">Скопировать все</button>
  <button onclick="clearAll()">Очистить</button>
</div>
<script>
  const W = {w}, H = {h};
  const img = document.getElementById('img');
  const list = document.getElementById('list');
  const pts = [];

  img.addEventListener('load', function() {{
    console.log('Карта загружена: ' + img.naturalWidth + '×' + img.naturalHeight);
  }});
  img.addEventListener('error', function() {{
    alert('Карта не загрузилась. Проверьте файл {img_file} рядом с HTML.');
  }});

  img.addEventListener('click', function(e) {{
    const rect = img.getBoundingClientRect();
    const x = Math.round((e.clientX - rect.left) * W / img.width);
    const y = Math.round((e.clientY - rect.top) * H / img.height);
    pts.push([x, y]);
    const div = document.createElement('div');
    div.className = 'pt';
    div.textContent = '#' + pts.length + ': ' + x + ',' + y;
    list.appendChild(div);
    // рисуем точку поверх карты
    const marker = document.createElement('div');
    marker.style.cssText = 'position:absolute;width:12px;height:12px;' +
      'background:red;border:2px solid yellow;border-radius:50%;' +
      'pointer-events:none;transform:translate(-50%,-50%);';
    marker.style.left = e.clientX - rect.left + 'px';
    marker.style.top = e.clientY - rect.top + 'px';
    marker.dataset.pid = pts.length;
    document.getElementById('wrap').appendChild(marker);
  }});

  function copyAll() {{
    if (pts.length === 0) {{ alert('Нет точек'); return; }}
    const text = pts.map(function(p) {{ return p.join(','); }}).join('\\n');
    navigator.clipboard.writeText(text).then(function() {{
      alert('Скопировано:\\n' + text);
    }});
  }}
  function clearAll() {{
    pts.length = 0;
    list.innerHTML = '';
    document.querySelectorAll('#wrap div').forEach(function(d) {{ d.remove(); }});
  }}
</script>
</body>
</html>"""

    out_html = f"{stem}_pick.html"
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\nHTML сохранён: {out_html}")
    print(f"Картинка: {img_file}")
    print(f"\nОткройте {out_html} → правой кнопкой → 'Open Preview'.")

def main():
    parser = argparse.ArgumentParser(description="Анализатор топографических карт")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # --- info ---
    p1 = sub.add_parser("info", help="Анализ карты: высоты, линии, сетка")
    p1.add_argument("--file", required=True)
    p1.add_argument("--grid-step", type=int, default=100)
    p1.add_argument("--save-json")
    p1.set_defaults(func=cmd_info)

    # --- measure ---
    p2 = sub.add_parser("measure", help="Измерить расстояние")
    p2.add_argument("--p1", required=True, help="x,y первой точки")
    p2.add_argument("--p2", required=True, help="x,y второй точки")
    p2.add_argument("--file", help="Файл для визуализации (опционально)")
    p2.add_argument("--meters-per-px", type=float)
    p2.add_argument("--scale", type=int, help="Знаменатель масштаба: 1:2000 → 2000")
    p2.add_argument("--dpi", type=int, help="DPI скана/PDF: 200, 300, 600")
    p2.add_argument("--real", type=float, help="Известное расстояние в метрах")
    p2.add_argument("--real-from", help="x,y опорной точки 1")
    p2.add_argument("--real-to", help="x,y опорной точки 2")
    p2.set_defaults(func=cmd_measure)

    # --- pixels ---
    p3 = sub.add_parser("pixels", help="Расстояние в пикселях + расчёт м/px")
    p3.add_argument("--p1", required=True, help="x,y первой точки")
    p3.add_argument("--p2", required=True, help="x,y второй точки")
    p3.add_argument("--real", type=float, help="Реальное расстояние (м), если известно")
    p3.set_defaults(func=cmd_pixels)

    # --- objects ---
    p4 = sub.add_parser("objects", help="Найти объекты и их координаты")
    p4.add_argument("--file", required=True)
    p4.add_argument("--min-area", type=float, default=20.0,
                    help="Минимальная площадь объекта (px²)")
    p4.add_argument("--limit", type=int, default=50,
                    help="Сколько объектов выводить")
    p4.add_argument("--save-json")
    p4.set_defaults(func=cmd_objects)

    # --- pick ---
    p5 = sub.add_parser("pick", help="HTML-страница для клика по карте")
    p5.add_argument("--file", required=True)
    p5.set_defaults(func=cmd_pick)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()