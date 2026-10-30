import streamlit as st
import tempfile
import os
import time
import threading
import json
import io
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import fitz  # PyMuPDF
import plotly.graph_objects as go
from scipy.interpolate import griddata
from streamlit_image_coordinates import streamlit_image_coordinates

from topo_ai import load_image, extract_heights

st.set_page_config(page_title="Topo AI", layout="wide")

# ============== Вспомогательные функции ==============
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
                base = doc.extract_image(images[0][0])
                img_w = base.get("width")
                if img_w and page_w_in > 0:
                    eff = int(round(img_w / page_w_in))
                    if 72 <= eff <= 1200:
                        doc.close()
                        return eff
            doc.close()
        except Exception:
            pass
    return 300


def resize_for_ocr(image, max_side=1500):
    h, w = image.shape[:2]
    if max(h, w) <= max_side:
        return image, 1.0
    scale = max_side / max(h, w)
    return cv2.resize(image, (int(w * scale), int(h * scale)),
                      interpolation=cv2.INTER_AREA), scale


def snap_to_black_dot(img_bgr, x, y, max_dist=60):
    h, w = img_bgr.shape[:2]
    x_min = max(0, x - max_dist)
    x_max = min(w, x + max_dist)
    y_min = max(0, y - max_dist)
    y_max = min(h, y + max_dist)

    patch = img_bgr[y_min:y_max, x_min:x_max]
    if patch.size == 0:
        return x, y

    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_dot = None
    min_dist = float('inf')

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 2 < area < 80:
            bx, by, bw, bh = cv2.boundingRect(cnt)
            aspect_ratio = float(bw) / bh if bh > 0 else 0
            if 0.5 <= aspect_ratio <= 2.0 and bw <= 12 and bh <= 12:
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"]) + x_min
                    cy = int(M["m01"] / M["m00"]) + y_min
                    dist = np.sqrt((cx - x)**2 + (cy - y)**2)
                    if dist < min_dist:
                        min_dist = dist
                        best_dot = (cx, cy)

    if best_dot:
        return best_dot[0], best_dot[1]
    return x, y


def to_real_coords(rows, mpp, anchor=None):
    out = []
    for r in rows:
        if anchor is None:
            X = float(r["x_px"]) * mpp
            Y = -float(r["y_px"]) * mpp
        else:
            X = anchor["X_real"] + (float(r["x_px"]) - anchor["x_px"]) * mpp
            Y = anchor["Y_real"] - (float(r["y_px"]) - anchor["y_px"]) * mpp
        out.append({"id": int(r["id"]), "x": X, "y": Y,
                    "value": float(r["value"]),
                    "source": r.get("source", "ocr")})
    return out


def interpolate_surface(points, grid_size=80):
    if len(points) < 3:
        return None
    x = np.array([p["x"] for p in points], dtype=float)
    y = np.array([p["y"] for p in points], dtype=float)
    z = np.array([p["value"] for p in points], dtype=float)
    xi = np.linspace(min(x), max(x), grid_size)
    yi = np.linspace(min(y), max(y), grid_size)
    xi_g, yi_g = np.meshgrid(xi, yi)
    zi = None
    for method in ("cubic", "linear", "nearest"):
        try:
            zi_try = griddata((x, y), z, (xi_g, yi_g), method=method)
            if zi_try is not None and not np.all(np.isnan(zi_try)):
                zi = zi_try
                break
        except Exception:
            continue
    if zi is None:
        return None
    if np.any(np.isnan(zi)):
        zi_nn = griddata((x, y), z, (xi_g, yi_g), method="nearest")
        zi = np.where(np.isnan(zi), zi_nn, zi)
    return xi, yi, zi


def to_geojson(points):
    feats = []
    for p in points:
        feats.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [p["x"], p["y"]]},
            "properties": {"id": p["id"], "height": p["value"]},
        })
    return {"type": "FeatureCollection", "features": feats}


# ============== Инициализация состояния ==============
def _init(key, value):
    if key not in st.session_state:
        st.session_state[key] = value


_init("points_df", pd.DataFrame(
    columns=["id", "value", "x_px", "y_px", "confidence", "source"]))
_init("img_bgr", None)
_init("mpp", 1.0)
_init("anchor", None)
_init("point_counter", 0)
_init("click_A", None)
_init("click_B", None)
_init("inspect_info", None)
_init("map2d_last_click", None)
_init("manual_click", None)
_init("ocr_running", False)


# ============== Sidebar ==============
st.sidebar.header("Масштаб карты")
scale_denom = st.sidebar.number_input(
    "Масштаб 1:N", min_value=100, value=2000, step=100
)

st.sidebar.header("Привязка (опционально)")
use_anchor = st.sidebar.checkbox("Привязать к кресту")
if use_anchor:
    st.sidebar.markdown("**Пиксельные координаты креста:**")
    ax_px = st.sidebar.number_input("X, px", min_value=0, value=0, step=1)
    ay_px = st.sidebar.number_input("Y, px", min_value=0, value=0, step=1)
    st.sidebar.markdown("**Реальные координаты креста:**")
    aX = st.sidebar.number_input("X, м", value=0.0, step=0.01, format="%.2f")
    aY = st.sidebar.number_input("Y, м", value=0.0, step=0.01, format="%.2f")
    st.session_state.anchor = {
        "x_px": int(ax_px), "y_px": int(ay_px),
        "X_real": float(aX), "Y_real": float(aY),
    }
else:
    st.session_state.anchor = None

if st.session_state.ocr_running:
    st.sidebar.warning("⏳ **OCR работает...** Это может занять 30–90 секунд. "
                       "Не закрывайте вкладку.")
else:
    st.sidebar.info("💡 OCR запускается по кнопке во вкладке **🤖 OCR**.")


# ============== Main ==============
st.title("Topo AI — анализ топографических карт")
st.markdown(
    "Загрузите карту, распознайте или добавьте точки вручную, "
    "кликните по карте для осмотра точек и измерения расстояний. "
    "Выгружайте координаты в CSV, Excel и GeoJSON (QGIS)."
)

uploaded = st.file_uploader(
    "Файл карты (PNG, JPG, PDF)", type=["png", "jpg", "jpeg", "pdf"]
)

if uploaded:
    ext = os.path.splitext(uploaded.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as f:
        f.write(uploaded.getvalue())
        tmp_path = f.name

    detected_dpi = auto_detect_dpi(tmp_path)
    try:
        st.session_state.img_bgr = load_image(tmp_path, dpi=detected_dpi)
    except Exception as e:
        st.error(f"Ошибка загрузки: {e}")
        st.stop()

    st.session_state.mpp = (scale_denom / 100.0) * 2.54 / detected_dpi
    st.info(
        f"DPI = **{detected_dpi}**. Масштаб 1:{scale_denom} → "
        f"**{st.session_state.mpp:.6f} м/px**."
    )
    try:
        os.unlink(tmp_path)
    except Exception:
        pass

if st.session_state.img_bgr is not None:
    img = st.session_state.img_bgr
    mpp = st.session_state.mpp

    tab_ocr, tab_manual, tab_table, tab_2d, tab_3d, tab_export = st.tabs([
        "🤖 OCR", "🖱️ Ручная разметка", "📋 Таблица",
        "🗺️ 2D-карта", "🏔️ 3D-рельеф", "💾 Экспорт",
    ])

    # ============ TAB OCR ============
    with tab_ocr:
        st.markdown(
            "Нажмите кнопку — EasyOCR попробует найти числовые отметки высот. "
            "**Прогресс отображается ниже и в боковой панели.**"
        )
        if st.button("🚀 Запустить OCR", type="primary",
                     disabled=st.session_state.ocr_running):
            st.session_state.ocr_running = True

            ocr_img, scale = resize_for_ocr(img, 1500)
            result_holder = {"done": False, "result": [], "error": None}
            started_at = time.time()

            def _ocr():
                try:
                    result_holder["result"] = extract_heights(ocr_img)
                except Exception as e:
                    result_holder["error"] = e
                finally:
                    result_holder["done"] = True

            threading.Thread(target=_ocr, daemon=True).start()

            with st.status("OCR выполняется...", expanded=True) as status:
                st.write("⏱️ **Этап 1/3:** Инициализация OCR...")
                progress = st.progress(0, text="Подготовка...")

                p = 0
                last_stage = 0
                while not result_holder["done"]:
                    elapsed = time.time() - started_at
                    p = min(95, p + 1)

                    if elapsed < 5:
                        stage_text = "Инициализация OCR..."
                        stage_pct = 5
                    elif elapsed < 15:
                        stage_text = "Загрузка моделей EasyOCR..."
                        stage_pct = 15
                    elif elapsed < 30:
                        stage_text = "Распознавание текста..."
                        stage_pct = 40
                    else:
                        stage_text = "Обработка результатов..."
                        stage_pct = 75

                    progress.progress(p, text=f"{stage_text} ({elapsed:.0f} с)")

                    if last_stage != stage_pct:
                        st.write(f"⏱️ **Этап:** {stage_text}")
                        last_stage = stage_pct

                    time.sleep(0.1)

                progress.progress(100, text="OCR завершён")
                elapsed_total = time.time() - started_at

                if result_holder["error"]:
                    status.update(label="❌ OCR завершился с ошибкой", state="error")
                    st.error(f"Ошибка OCR: {result_holder['error']}")
                else:
                    found_raw = len(result_holder["result"])
                    st.write(f"✅ **Этап 3/3:** Найдено текстовых блоков: {found_raw}")

                    rows = []
                    for h in result_holder["result"]:
                        orig_x = int(h["x_px"] / scale)
                        orig_y = int(h["y_px"] / scale)

                        snapped_x, snapped_y = snap_to_black_dot(img, orig_x, orig_y, max_dist=60)

                        st.session_state.point_counter += 1
                        rows.append({
                            "id": st.session_state.point_counter,
                            "value": round(h["value"], 2),
                            "x_px": snapped_x,
                            "y_px": snapped_y,
                            "confidence": h.get("confidence", 1.0),
                            "source": "ocr",
                        })

                    if rows:
                        st.session_state.points_df = pd.concat(
                            [st.session_state.points_df, pd.DataFrame(rows)],
                            ignore_index=True,
                        )

                    st.write(f"✅ **Добавлено точек-высот:** {len(rows)}")
                    st.write(f"⏱️ **Общее время:** {elapsed_total:.1f} с")
                    status.update(
                        label=f"✅ OCR завершён за {elapsed_total:.1f} с. Найдено {len(rows)} высот.",
                        state="complete",
                    )

            st.session_state.ocr_running = False
            st.rerun()

        if not st.session_state.points_df.empty:
            st.subheader("Текущие точки")
            vis = img.copy()
            for _, r in st.session_state.points_df.iterrows():
                cv2.circle(vis, (int(r["x_px"]), int(r["y_px"])), 4, (0, 0, 255), -1)
                cv2.putText(vis, f"{r['value']:.2f}",
                            (int(r["x_px"]) + 8, int(r["y_px"]) - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            st.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB),
                     width="stretch",
                     caption=f"Точек: {len(st.session_state.points_df)}")

    # ============ TAB MANUAL ============
    with tab_manual:
        st.markdown(
            "**Кликните на карту** — координаты появятся ниже. "
            "Введите высоту и нажмите «Добавить»."
        )

        col1, col2 = st.columns([1, 1])
        with col1:
            zoom_manual = st.slider("🔍 Масштаб отображения карты (Zoom)", 0.1, 1.0, 0.5, 0.1)
        with col2:
            delete_mode = st.checkbox("🗑️ Режим удаления (клик по точке удалит её)")

        df_manual = st.session_state.points_df.dropna(subset=["value", "x_px", "y_px"])

        # Масштабируем карту для быстрой работы
        disp_h = int(img.shape[0] * zoom_manual)
        disp_w = int(img.shape[1] * zoom_manual)
        disp_img = cv2.resize(img, (disp_w, disp_h), interpolation=cv2.INTER_AREA)

        vis_manual = disp_img.copy()
        for _, r in df_manual.iterrows():
            x = int(r["x_px"] * zoom_manual)
            y = int(r["y_px"] * zoom_manual)
            cv2.circle(vis_manual, (x, y), 4, (255, 0, 0), -1)
            cv2.putText(vis_manual, f"{r['value']:.2f}", (x + 8, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        if st.session_state.manual_click and not delete_mode:
            rcx, rcy = st.session_state.manual_click
            cv2.drawMarker(vis_manual, (int(rcx * zoom_manual), int(rcy * zoom_manual)), (0, 0, 255),
                           markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)

        vis_pil = Image.fromarray(cv2.cvtColor(vis_manual, cv2.COLOR_BGR2RGB))
        
        # Возвращаем стандартный вывод без контейнеров, чтобы клики работали
        click = streamlit_image_coordinates(vis_pil, key="manual_click_widget")

        if click is not None:
            # Переводим координаты клика обратно в реальные координаты оригинальной карты
            real_x = int(click["x"] / zoom_manual)
            real_y = int(click["y"] / zoom_manual)
            coords = (real_x, real_y)

            if coords != st.session_state.manual_click:
                if delete_mode:
                    if not df_manual.empty:
                        dists = np.sqrt(
                            (df_manual["x_px"].astype(float) - coords[0]) ** 2 +
                            (df_manual["y_px"].astype(float) - coords[1]) ** 2
                        )
                        if dists.min() < (30 / zoom_manual):
                            idx_to_drop = dists.idxmin()
                            st.session_state.points_df = st.session_state.points_df.drop(idx_to_drop).reset_index(drop=True)
                            st.session_state.manual_click = None
                            st.rerun()
                else:
                    st.session_state.manual_click = coords
                    st.rerun()

        st.caption(f"Точек уже добавлено: **{len(df_manual)}**")

        if st.session_state.manual_click and not delete_mode:
            rcx, rcy = st.session_state.manual_click
            st.info(f"Координаты клика: **({rcx}, {rcy})** px. Введите высоту и нажмите «Добавить».")
            with st.form("add_point_form"):
                height = st.number_input("Высота, м", value=200.0, step=0.01, format="%.2f")
                col_a, col_b = st.columns(2)
                with col_a:
                    x_edit = st.number_input("X, px", value=rcx, step=1)
                with col_b:
                    y_edit = st.number_input("Y, px", value=rcy, step=1)
                if st.form_submit_button("Добавить точку", type="primary"):
                    st.session_state.point_counter += 1
                    new = pd.DataFrame([{
                        "id": st.session_state.point_counter,
                        "value": round(float(height), 2),
                        "x_px": int(x_edit),
                        "y_px": int(y_edit),
                        "confidence": 1.0,
                        "source": "manual",
                    }])
                    st.session_state.points_df = pd.concat(
                        [st.session_state.points_df, new], ignore_index=True
                    )
                    st.session_state.manual_click = None
                    st.rerun()

    # ============ TAB TABLE ============
    with tab_table:
        st.markdown("Вы можете редактировать значения прямо здесь. Для удаления выделите строку и нажмите Delete.")
        edited = st.data_editor(
            st.session_state.points_df,
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "id": st.column_config.NumberColumn("ID", disabled=True),
                "value": st.column_config.NumberColumn("Высота, м", format="%.2f"),
                "x_px": st.column_config.NumberColumn("X, px", format="%d"),
                "y_px": st.column_config.NumberColumn("Y, px", format="%d"),
                "confidence": st.column_config.NumberColumn("Conf", format="%.2f", disabled=True),
                "source": st.column_config.TextColumn("Источник", disabled=True),
            },
        )
        st.session_state.points_df = edited
        if st.button("Очистить все точки"):
            st.session_state.points_df = st.session_state.points_df.iloc[0:0]
            st.session_state.point_counter = 0
            st.rerun()

    # ============ TAB 2D MAP ============
    with tab_2d:
        df = st.session_state.points_df.dropna(subset=["value", "x_px", "y_px"])

        st.markdown(
            "**Кликните по карте:** в режиме «Осмотр» — увидите ближайшую точку. "
            "В режиме «Измерение» — кликните две точки, и появится расстояние."
        )

        col1, col2 = st.columns([1, 1])
        with col1:
            zoom_2d = st.slider("🔍 Масштаб отображения карты (Zoom)", 0.1, 1.0, 0.5, 0.1, key="zoom_2d")
        with col2:
            mode = st.radio(
                "Режим клика",
                ["🔍 Осмотр точек", "📏 Измерение расстояния"],
                horizontal=True, key="map2d_mode",
            )

        disp_h = int(img.shape[0] * zoom_2d)
        disp_w = int(img.shape[1] * zoom_2d)
        disp_img = cv2.resize(img, (disp_w, disp_h), interpolation=cv2.INTER_AREA)

        vis = disp_img.copy()
        for _, r in df.iterrows():
            x = int(r["x_px"] * zoom_2d)
            y = int(r["y_px"] * zoom_2d)
            cv2.circle(vis, (x, y), 4, (0, 0, 255), -1)
            cv2.putText(vis, f"{r['value']:.2f}", (x + 8, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        if mode == "📏 Измерение расстояния":
            if st.session_state.click_A:
                ax = int(st.session_state.click_A[0] * zoom_2d)
                ay = int(st.session_state.click_A[1] * zoom_2d)
                cv2.circle(vis, (ax, ay), 10, (255, 0, 0), 2)
                cv2.putText(vis, "A", (ax + 12, ay), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            if st.session_state.click_B:
                bx = int(st.session_state.click_B[0] * zoom_2d)
                by = int(st.session_state.click_B[1] * zoom_2d)
                cv2.circle(vis, (bx, by), 10, (255, 0, 0), 2)
                cv2.putText(vis, "B", (bx + 12, by), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            if st.session_state.click_A and st.session_state.click_B:
                ax = int(st.session_state.click_A[0] * zoom_2d)
                ay = int(st.session_state.click_A[1] * zoom_2d)
                bx = int(st.session_state.click_B[0] * zoom_2d)
                by = int(st.session_state.click_B[1] * zoom_2d)
                cv2.line(vis, (ax, ay), (bx, by), (255, 0, 0), 2)

        vis_pil = Image.fromarray(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        click = streamlit_image_coordinates(vis_pil, key="map2d_click_widget")

        if click is not None:
            real_x = int(click["x"] / zoom_2d)
            real_y = int(click["y"] / zoom_2d)
            coords = (real_x, real_y)

            if coords != st.session_state.map2d_last_click:
                st.session_state.map2d_last_click = coords

                if mode == "🔍 Осмотр точек":
                    if not df.empty:
                        dists = np.sqrt(
                            (df["x_px"].astype(float) - coords[0]) ** 2 +
                            (df["y_px"].astype(float) - coords[1]) ** 2
                        )
                        idx = dists.idxmin()
                        nearest = df.loc[idx].to_dict()
                        real = to_real_coords([nearest], mpp, st.session_state.anchor)[0]
                        st.session_state.inspect_info = {
                            "id": int(nearest["id"]),
                            "value": float(nearest["value"]),
                            "x_px": int(nearest["x_px"]),
                            "y_px": int(nearest["y_px"]),
                            "X_real": real["x"],
                            "Y_real": real["y"],
                            "dist_px": float(dists.min()),
                        }
                else:
                    if st.session_state.click_A is None:
                        st.session_state.click_A = coords
                    elif st.session_state.click_B is None:
                        st.session_state.click_B = coords
                    else:
                        st.session_state.click_A = coords
                        st.session_state.click_B = None
                st.rerun()

        if mode == "🔍 Осмотр точек":
            if st.session_state.inspect_info:
                info = st.session_state.inspect_info
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("ID", info["id"])
                c2.metric("Высота, м", f"{info['value']:.2f}")
                c3.metric("X, px", info["x_px"])
                c4.metric("Y, px", info["y_px"])
                st.caption(
                    f"Реальные координаты: X = {info['X_real']:.2f} м, "
                    f"Y = {info['Y_real']:.2f} м. "
                    f"От клика до точки: {info['dist_px']:.0f} px."
                )
            else:
                st.info("Кликните по карте рядом с точкой — увидите её данные.")
        else:
            if st.session_state.click_A and st.session_state.click_B:
                ax, ay = st.session_state.click_A
                bx, by = st.session_state.click_B
                dx = (bx - ax) * mpp
                dy = -(by - ay) * mpp
                dist = np.sqrt(dx ** 2 + dy ** 2)
                c1, c2, c3 = st.columns(3)
                c1.metric("ΔX, м", f"{dx:+.2f}")
                c2.metric("ΔY, м", f"{dy:+.2f}")
                c3.metric("Расстояние, м", f"{dist:.2f}")
                st.caption(f"A = ({ax}, {ay}) px, B = ({bx}, {by}) px. Ещё один клик начнёт новое измерение.")
            elif st.session_state.click_A:
                st.info("Точка A задана. Кликните вторую точку (B).")
            else:
                st.info("Кликните первую точку (A).")

            if st.button("Сбросить измерение"):
                st.session_state.click_A = None
                st.session_state.click_B = None
                st.session_state.map2d_last_click = None
                st.rerun()

    # ============ TAB 3D ============
    with tab_3d:
        df = st.session_state.points_df.dropna(subset=["value", "x_px", "y_px"])
        if len(df) < 3:
            st.warning("Для 3D-модели нужно минимум 3 точки")
        else:
            real = to_real_coords(df.to_dict("records"), mpp, st.session_state.anchor)
            surface = interpolate_surface(real, grid_size=80)
            if surface is None:
                st.error("Не удалось построить поверхность")
            else:
                xi, yi, zi = surface
                fig3d = go.Figure(data=[
                    go.Surface(x=xi, y=yi, z=zi, colorscale="Jet",
                               colorbar=dict(title="Высота, м"),
                               contours={"z": {"show": True, "usecolormap": True}}),
                    go.Scatter3d(x=[p["x"] for p in real], y=[p["y"] for p in real],
                                 z=[p["value"] for p in real], mode="markers",
                                 marker=dict(size=3, color="black"), name="Точки"),
                ])
                fig3d.update_layout(
                    title="3D-модель рельефа",
                    scene=dict(xaxis_title="X, м", yaxis_title="Y, м", zaxis_title="Высота, м", aspectmode="data"),
                    height=700,
                )
                st.plotly_chart(fig3d, use_container_width=True)

    # ============ TAB EXPORT ============
    with tab_export:
        df = st.session_state.points_df.dropna(subset=["value", "x_px", "y_px"])
        if df.empty:
            st.warning("Нет данных для экспорта")
        else:
            real = to_real_coords(df.to_dict("records"), mpp, st.session_state.anchor)
            df_real = pd.DataFrame(real)
            df_real = df_real[["id", "x", "y", "value", "source"]]
            df_real.columns = ["ID", "X_м", "Y_м", "Высота_м", "Источник"]

            st.subheader("CSV")
            st.download_button(
                "⬇️ Скачать CSV",
                data=df_real.to_csv(index=False).encode("utf-8-sig"),
                file_name="topo_points.csv",
                mime="text/csv",
            )

            st.subheader("Excel")
            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine="openpyxl") as writer:
                df_real.to_excel(writer, index=False, sheet_name="Точки")
            st.download_button(
                "⬇️ Скачать Excel",
                data=buf.getvalue(),
                file_name="topo_points.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

            st.subheader("GeoJSON (для QGIS)")
            geojson_str = json.dumps(to_geojson(real), ensure_ascii=False, indent=2)
            st.download_button(
                "⬇️ Скачать GeoJSON",
                data=geojson_str.encode("utf-8"),
                file_name="topo_points.geojson",
                mime="application/geo+json",
            )
            st.caption(
                "В QGIS: Слой → Добавить векторный слой → выбрать файл. "
                "Координаты локальные (м от привязки), при импорте выберите «Без CRS»."
            )