import streamlit as st
import tempfile
import os
import time
import threading
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import fitz  # PyMuPDF
from topo_ai import load_image, extract_heights
from heatmap_generator import create_heatmap

st.set_page_config(page_title="Анализатор топографических карт", layout="wide")


# ---------- Автоопределение DPI ----------
def auto_detect_dpi(image_path):
    path = Path(image_path)
    ext = path.suffix.lower()

    if ext in (".jpg", ".jpeg", ".png", ".tif", ".tiff"):
        try:
            with Image.open(image_path) as im:
                dpi = im.info.get("dpi")
                if dpi and dpi[0] and dpi[0] > 50:
                    return int(dpi[0])
        except Exception:
            pass

    if ext == ".pdf":
        try:
            doc = fitz.open(str(path))
            page = doc.load_page(0)
            page_w_in = page.rect.width / 72.0
            images = page.get_images(full=True)
            if images:
                xref = images[0][0]
                base = doc.extract_image(xref)
                img_w = base.get("width")
                if img_w and page_w_in > 0:
                    eff_dpi = int(round(img_w / page_w_in))
                    if 72 <= eff_dpi <= 1200:
                        doc.close()
                        return eff_dpi
            doc.close()
        except Exception:
            pass
    return 300


# ---------- Фильтрация горизонталей ----------
def filter_contours_by_shape(binary_mask, min_area=50, max_area=50000,
                             max_rectangularity=0.8, min_vertices=5):
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    filtered = np.zeros_like(binary_mask)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h == 0:
            continue
        if area / (w * h) > max_rectangularity:
            continue
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        if len(approx) < min_vertices:
            continue
        cv2.drawContours(filtered, [cnt], -1, 255, thickness=cv2.FILLED)
    return filtered


def resize_for_ocr(image, max_side=1500):
    h, w = image.shape[:2]
    if max(h, w) <= max_side:
        return image, 1.0
    scale = max_side / max(h, w)
    resized = cv2.resize(image, (int(w * scale), int(h * scale)),
                         interpolation=cv2.INTER_AREA)
    return resized, scale


# ---------- Пересчёт в реальные координаты ----------
def to_real_coords(heights, mpp, anchor=None):
    """
    heights: [{'x_px': .., 'y_px': .., 'value': ..}, ...]
    mpp: метров на пиксель
    anchor: {'x_px': .., 'y_px': .., 'X_real': .., 'Y_real': ..} или None
    Возвращает: [{'x': метры, 'y': метры, 'value': ..}, ...]
    """
    result = []
    for h in heights:
        if anchor is None:
            X = h["x_px"] * mpp
            Y = -h["y_px"] * mpp  # инверсия: север сверху
        else:
            X = anchor["X_real"] + (h["x_px"] - anchor["x_px"]) * mpp
            Y = anchor["Y_real"] - (h["y_px"] - anchor["y_px"]) * mpp
        result.append({"x": X, "y": Y, "value": h["value"]})
    return result


# ========== БОКОВАЯ ПАНЕЛЬ ==========
st.sidebar.header("Масштаб карты")
scale_denom = st.sidebar.number_input(
    "Масштаб 1:N", min_value=100, value=2000, step=100,
    help="Например, для карты 1:1000 введите 1000"
)

st.sidebar.header("Привязка координат (необязательно)")
use_anchor = st.sidebar.checkbox(
    "Привязать к известному кресту (геодезическому пункту)",
    help="Если известны реальные координаты одного пункта на карте, "
         "можно привязать все точки к ним. Карта должна быть ориентирована по северу."
)

anchor = None
if use_anchor:
    st.sidebar.markdown("**Пиксельные координаты креста на изображении:**")
    anchor_x_px = st.sidebar.number_input(
        "X креста (пиксели)", min_value=0, value=0, step=1
    )
    anchor_y_px = st.sidebar.number_input(
        "Y креста (пиксели)", min_value=0, value=0, step=1
    )

    st.sidebar.markdown("**Реальные координаты этого креста (метры):**")
    anchor_X_real = st.sidebar.number_input(
        "X реальное (м)", value=0.0, step=0.01, format="%.2f"
    )
    anchor_Y_real = st.sidebar.number_input(
        "Y реальное (м)", value=0.0, step=0.01, format="%.2f"
    )

    anchor = {
        "x_px": anchor_x_px,
        "y_px": anchor_y_px,
        "X_real": anchor_X_real,
        "Y_real": anchor_Y_real,
    }


# ========== ОСНОВНОЕ ОКНО ==========
st.title("Анализатор топографических карт")
st.markdown(
    "Загрузите карту. Система выделит горизонтали, распознает высоты через OCR, "
    "даст возможность **отредактировать найденные точки** и построит тепловую карту "
    "в реальных координатах. Если известны координаты одного креста на карте — "
    "включите привязку в боковой панели."
)

uploaded_file = st.file_uploader(
    "Выберите файл карты (PNG, JPG, PDF)",
    type=["png", "jpg", "jpeg", "pdf"]
)

if uploaded_file is not None:
    ext = os.path.splitext(uploaded_file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        image_path = tmp_file.name

    st.success(f"Файл загружен: {uploaded_file.name}")

    detected_dpi = auto_detect_dpi(image_path)
    try:
        image = load_image(image_path, dpi=detected_dpi)
    except Exception as e:
        st.error(f"Не удалось загрузить изображение: {e}")
        st.stop()

    meters_per_px = (scale_denom / 100.0) * 2.54 / detected_dpi

    st.info(
        f"DPI определён: **{detected_dpi}**. "
        f"Масштаб 1:{scale_denom} → **{meters_per_px:.6f} м/px**."
    )

    # Отрисовка креста на исходной карте (для проверки)
    vis_anchor = image.copy()
    if anchor is not None:
        cv2.drawMarker(
            vis_anchor, (anchor["x_px"], anchor["y_px"]),
            (0, 0, 255), markerType=cv2.MARKER_CROSS,
            markerSize=30, thickness=3
        )
        cv2.putText(
            vis_anchor,
            f"({anchor['X_real']:.1f}, {anchor['Y_real']:.1f})",
            (anchor["x_px"] + 15, anchor["y_px"] - 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2
        )

    st.subheader("Исходная карта")
    st.image(
        cv2.cvtColor(vis_anchor, cv2.COLOR_BGR2RGB),
        width='stretch',
        caption="Красный крест — точка привязки (если задана)"
    )

    # Горизонтали
    st.subheader("Выделение линий (горизонталей)")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )
    filtered_mask = filter_contours_by_shape(thresh)
    if int(np.count_nonzero(filtered_mask)) > 0:
        st.image(filtered_mask, caption="Маска горизонталей", width='stretch')
    else:
        st.warning("Горизонтали не обнаружены — карта будет построена только по OCR.")

    # ---------- OCR ----------
    st.subheader("Распознавание высот")
    if "heights_df" not in st.session_state:
        ocr_image, scale = resize_for_ocr(image, max_side=1500)
        st.caption(f"Размер для OCR: {ocr_image.shape[1]}×{ocr_image.shape[0]}")

        progress_bar = st.progress(0, text="Инициализация OCR...")
        result_holder = {"heights": [], "error": None, "done": False}

        def _run_ocr():
            try:
                result_holder["heights"] = extract_heights(ocr_image)
            except Exception as e:
                result_holder["error"] = e
            finally:
                result_holder["done"] = True

        thread = threading.Thread(target=_run_ocr, daemon=True)
        thread.start()

        pct = 0
        while not result_holder["done"]:
            pct = min(95, pct + 1)
            progress_bar.progress(pct, text=f"Распознавание... {pct}%")
            time.sleep(0.1)

        progress_bar.progress(100, text="OCR завершён")
        heights_raw = result_holder["heights"]
        if result_holder["error"]:
            st.error(f"Ошибка OCR: {result_holder['error']}")
            heights_raw = []

        # Пересчёт координат обратно к исходному изображению
        if scale != 1.0:
            for h in heights_raw:
                h["x_px"] = int(h["x_px"] / scale)
                h["y_px"] = int(h["y_px"] / scale)

        st.session_state["heights_df"] = pd.DataFrame(heights_raw)

    # ---------- Редактор ----------
    st.markdown(
        "**Отредактируйте найденные точки:** "
        "можно исправить значение, координаты, удалить ошибочные строки "
        "или добавить новую точку кнопкой `+` внизу таблицы."
    )

    edited_df = st.data_editor(
        st.session_state["heights_df"],
        num_rows="dynamic",
        column_config={
            "value": st.column_config.NumberColumn(
                "Высота, м", format="%.2f", min_value=0.0, max_value=9000.0
            ),
            "x_px": st.column_config.NumberColumn(
                "X (пиксели)", format="%d", min_value=0
            ),
            "y_px": st.column_config.NumberColumn(
                "Y (пиксели)", format="%d", min_value=0
            ),
            "confidence": st.column_config.NumberColumn(
                "Уверенность", format="%.2f", disabled=True
            ),
        },
        key="heights_editor",
        use_container_width=True,
    )

    # Сохраняем изменения
    st.session_state["heights_df"] = edited_df

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Сбросить (вернуть OCR)"):
            del st.session_state["heights_df"]
            st.rerun()
    with col2:
        build_map = st.button("Построить тепловую карту", type="primary")

    # ---------- Построение карты ----------
    if build_map:
        clean = edited_df.dropna(subset=["value", "x_px", "y_px"])
        heights_list = clean.to_dict("records")
        st.write(f"Используется точек: **{len(heights_list)}**")

        if len(heights_list) >= 3:
            points_real = to_real_coords(heights_list, meters_per_px, anchor)

            st.subheader("Координаты точек в реальных метрах")
            df_real = pd.DataFrame(points_real)
            df_real.columns = ["X (м)", "Y (м)", "Высота, м"]
            st.dataframe(df_real, use_container_width=True)

            with st.spinner("Построение тепловой карты..."):
                html_path = create_heatmap(points_real,
                                           output_html="temp_heatmap.html")

            if html_path and os.path.exists(html_path):
                st.subheader("Тепловая карта высот")
                st.success(f"Карта построена по {len(points_real)} точкам.")
                st.iframe(Path(html_path), height=600)
            else:
                st.error("Не удалось построить тепловую карту (интерполяция не удалась).")
        elif len(heights_list) > 0:
            st.warning(
                f"Точек {len(heights_list)}. Для карты нужно минимум 3. "
                "Показываю найденные точки на изображении."
            )
            vis = image.copy()
            for h in heights_list:
                cv2.circle(vis, (int(h["x_px"]), int(h["y_px"])), 8, (0, 0, 255), -1)
                cv2.putText(
                    vis, f"{h['value']:.1f}",
                    (int(h["x_px"]) + 10, int(h["y_px"]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2
                )
            st.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), width='stretch')
        else:
            st.error("Нет точек для построения. Отредактируйте таблицу или сбросьте OCR.")

    try:
        os.unlink(image_path)
    except Exception:
        pass