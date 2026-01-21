import json
import os
import re
import time

from .deepseek_ocr import deepseek_ocr_image
from .logging_utils import log
from .text_utils import line_contains_any, normalize_for_match

QUESTIONNAIRE_FIELDS = [
    ("q1a", "cuestionario.1_recursos.a_pagina_web", "Pagina web"),
    ("q1b", "cuestionario.1_recursos.b_cuenta_bancaria", "Cuenta bancaria"),
    ("q1c", "cuestionario.1_recursos.c_encargado_comercio_exterior", "persona encargada comercio exterior"),
    ("q1d", "cuestionario.1_recursos.d_personal_ingles_idioma_objetivo", "idioma mercado objetivo"),
    ("q1e", "cuestionario.1_recursos.e_capital_para_demanda_exterior", "capital responder demandas"),
    ("q1f", "cuestionario.1_recursos.f_material_promocional", "material promocional"),
    ("q2b", "cuestionario.2_experiencia.b_mas_de_3_ciudades_importantes_py", "mas de 3 ciudades"),
    ("q2d", "cuestionario.2_experiencia.d_empresa_exporta.exporta", "la empresa exporta"),
    ("q3a", "cuestionario.3_producto.a_producto_servicio_propio_comercializado", "producto servicio propio"),
    ("q3b", "cuestionario.3_producto.b_capacidad_aumentar_produccion_ventas_internacionales", "capacidad aumentar produccion"),
    ("q3c", "cuestionario.3_producto.c_certificacion_internacional", "certificacion internacional"),
    ("q4a", "cuestionario.4_planificacion.a_presupuesto_anual", "presupuesto anual"),
    ("q4b", "cuestionario.4_planificacion.b_objetivos_metas_estrategias_mercado_local", "objetivos metas estrategias"),
    ("q4c", "cuestionario.4_planificacion.c_plan_exportacion", "plan de exportacion"),
    ("q5a1", "cuestionario.5_conocimiento_mercado_exterior.a_conoce_normas_y_entorno.normas_envase", "normas envase"),
    ("q5a2", "cuestionario.5_conocimiento_mercado_exterior.a_conoce_normas_y_entorno.normas_embalaje", "normas embalaje"),
    ("q5a3", "cuestionario.5_conocimiento_mercado_exterior.a_conoce_normas_y_entorno.normas_etiquetado", "normas etiquetado"),
    ("q5a4", "cuestionario.5_conocimiento_mercado_exterior.a_conoce_normas_y_entorno.estandares_calidad_y_certificaciones", "estandares calidad"),
    ("q5a5", "cuestionario.5_conocimiento_mercado_exterior.a_conoce_normas_y_entorno.aspectos_culturales", "aspectos culturales"),
    ("q5a6", "cuestionario.5_conocimiento_mercado_exterior.a_conoce_normas_y_entorno.competidores", "competidores"),
    ("q5b", "cuestionario.5_conocimiento_mercado_exterior.b_conoce_medios_transporte_y_costos", "medios de transporte"),
    ("q5c", "cuestionario.5_conocimiento_mercado_exterior.c_conoce_canales_comercializacion", "canales de comercializacion"),
    ("q6a", "cuestionario.6_barreras.a_conoce_barreras_arancelarias_no_arancelarias", "barreras arancelarias"),
    ("q6b", "cuestionario.6_barreras.b_conoce_acuerdos_que_benefician_producto", "acuerdos paraguay"),
    ("q6c", "cuestionario.6_barreras.c_conoce_regulaciones_internacionales", "regulaciones internacionales"),
    ("q7a", "cuestionario.7_precios.a_conoce_precios_mayoristas_minoristas_competencia", "precios mayoristas"),
    ("q7b", "cuestionario.7_precios.b_conoce_estructura_margenes_industria", "estructura margenes"),
    ("q8a", "cuestionario.8_desempeno_ambiental.a_tiene_habilitacion_ambiental", "habilitacion ambiental"),
]


def build_questionnaire_prompt():
    keys = [key for key, _, _ in QUESTIONNAIRE_FIELDS]
    lines = [
        "<image>",
        "<|grounding|>You are reading a survey table with YES/NO checkboxes.",
        "Left checkbox means 'si'. Right checkbox means 'no'.",
        "Return ONLY valid JSON.",
        "Use the short keys listed. Values: 'si', 'no', or '' if unclear.",
        "Example output:",
        "{",
        f"  \"{keys[0]}\": \"si\",",
        f"  \"{keys[1]}\": \"no\"",
        "}",
        "Keys:",
    ]
    for key, _, label in QUESTIONNAIRE_FIELDS:
        lines.append(f"{key}: {label}")
    return "\n".join(lines) + "\n"


def parse_yes_no_from_line(line):
    if not line:
        return None
    if "|" in line:
        parts = [part.strip() for part in line.split("|") if part.strip()]
        if len(parts) >= 3:
            yes_seg = parts[1]
            no_seg = parts[2]
            if re.search(r"\b(x|si|s[iA-]|sa-)\b", yes_seg, flags=re.IGNORECASE):
                return True
            if re.search(r"\b(x|no)\b", no_seg, flags=re.IGNORECASE):
                return False
    if re.search(r"\b(s[iA-]|si|sa-)\b", line, flags=re.IGNORECASE):
        if not re.search(r"\bno\b", line, flags=re.IGNORECASE):
            return True
    if re.search(r"\bno\b", line, flags=re.IGNORECASE):
        if not re.search(r"\b(s[iA-]|si|sa-)\b", line, flags=re.IGNORECASE):
            return False
    if re.search(r"\bsi\b.*\bx\b", line, flags=re.IGNORECASE):
        return True
    if re.search(r"\bno\b.*\bx\b", line, flags=re.IGNORECASE):
        return False
    return None


def normalize_yes_no_value(value):
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    text = normalize_for_match(value)
    if not text:
        return None
    if "si" in text or text == "x":
        return True
    if "no" in text:
        return False
    return None


def parse_questionnaire_response(text):
    cleaned = (text or "").strip()
    if not cleaned:
        return {}
    if "```" in cleaned:
        cleaned = cleaned.replace("```json", "").replace("```", "").strip()
    first = cleaned.find("{")
    last = cleaned.rfind("}")
    if first != -1 and last != -1 and last > first:
        cleaned = cleaned[first : last + 1]
    try:
        payload = json.loads(cleaned)
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        pass
    result = {}
    for line in cleaned.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip().strip('"').strip("'")
        value = value.strip().strip('"').strip("'")
        if key:
            result[key] = value
    return result


def set_nested_value(data, path, value):
    parts = path.split(".")
    ref = data
    for part in parts[:-1]:
        if part not in ref or not isinstance(ref[part], dict):
            ref[part] = {}
        ref = ref[part]
    ref[parts[-1]] = value


def get_nested_value(data, path):
    ref = data
    for part in path.split("."):
        if not isinstance(ref, dict) or part not in ref:
            return None
        ref = ref[part]
    return ref


def find_questionnaire_pages(text_pages):
    pages = []
    for page_num, text in text_pages:
        if line_contains_any(text, ["Cuestionario", "Marque con una X", "Marque con una"]):
            pages.append(page_num)
    return sorted(set(pages))


def _deskew_gray(gray):
    try:
        import cv2
        import numpy as np
    except ImportError:
        return gray

    edges = cv2.Canny(gray, 50, 150)
    coords = cv2.findNonZero(edges)
    if coords is None:
        return gray
    rect = cv2.minAreaRect(coords)
    angle = rect[-1]
    if angle < -45:
        angle += 90
    angle = -angle
    if abs(angle) < 0.5 or abs(angle) > 6:
        return gray
    height, width = gray.shape[:2]
    center = (width // 2, height // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        gray,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255,
    )


def detect_checkbox_rows(
    image,
    deskew=False,
    contrast=1.0,
    binarize=True,
    crop_left=0.0,
    debug_dir=None,
    page_num=None,
):
    try:
        import cv2
        import numpy as np
    except ImportError:
        return None

    rgb = image.convert("RGB")
    gray = cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2GRAY)
    if crop_left and crop_left > 0:
        left = int(gray.shape[1] * crop_left)
        gray = gray[:, left:]
    if deskew:
        gray = _deskew_gray(gray)
    if contrast and contrast != 1.0:
        gray = cv2.convertScaleAbs(gray, alpha=contrast, beta=0)
    height, width = gray.shape[:2]
    min_dim = min(width, height)
    min_size = max(8, int(min_dim * 0.007))
    max_size = int(min_dim * 0.04)
    right_region = 0 if crop_left and crop_left > 0 else int(width * 0.35)

    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    if binarize:
        thresh = cv2.adaptiveThreshold(
            blur,
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV,
            31,
            10,
        )
    else:
        _, thresh = cv2.threshold(
            blur,
            0,
            255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
        )
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w < min_size or h < min_size or w > max_size or h > max_size:
            continue
        ratio = w / float(h)
        if ratio < 0.7 or ratio > 1.3:
            continue
        if x < right_region:
            continue
        pad = max(1, int(min(w, h) * 0.2))
        x1 = x + pad
        y1 = y + pad
        x2 = x + w - pad
        y2 = y + h - pad
        if x2 <= x1 or y2 <= y1:
            continue
        inner = thresh[y1:y2, x1:x2]
        ink_ratio = float(inner.mean()) / 255.0
        boxes.append(
            {
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "checked": ink_ratio > 0.05,
            }
        )

    if debug_dir and page_num is not None:
        os.makedirs(debug_dir, exist_ok=True)
        base_name = f"checkbox_page_{page_num}"
        cv2.imwrite(os.path.join(debug_dir, f"{base_name}_gray.png"), gray)
        cv2.imwrite(os.path.join(debug_dir, f"{base_name}_thresh.png"), thresh)
        debug_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        for box in boxes:
            color = (0, 200, 0) if box["checked"] else (0, 0, 200)
            cv2.rectangle(
                debug_img,
                (box["x"], box["y"]),
                (box["x"] + box["w"], box["y"] + box["h"]),
                color,
                2,
            )
        cv2.imwrite(os.path.join(debug_dir, f"{base_name}_boxes.png"), debug_img)

    if not boxes:
        return []

    boxes.sort(key=lambda item: (item["y"], item["x"]))
    heights = sorted(box["h"] for box in boxes)
    median_h = heights[len(heights) // 2]
    row_tol = max(6, int(median_h * 0.9))
    rows = []
    for box in boxes:
        cy = box["y"] + box["h"] / 2.0
        if not rows or abs(cy - rows[-1]["cy"]) > row_tol:
            rows.append({"cy": cy, "boxes": [box]})
        else:
            rows[-1]["boxes"].append(box)

    results = []
    for row in rows:
        row_boxes = sorted(row["boxes"], key=lambda item: item["x"])
        if len(row_boxes) < 2:
            continue
        left = row_boxes[0]
        right = row_boxes[-1]
        results.append({"si": left["checked"], "no": right["checked"]})
    return results


def apply_checkbox_rows(data, rows):
    filled = 0
    for idx, row in enumerate(rows):
        if idx >= len(QUESTIONNAIRE_FIELDS):
            break
        _, path, _ = QUESTIONNAIRE_FIELDS[idx]
        value = None
        if row.get("si") and not row.get("no"):
            value = True
        elif row.get("no") and not row.get("si"):
            value = False
        if value is None:
            continue
        if path.endswith(".exporta"):
            value_to_set = "si" if value else "no"
        else:
            value_to_set = value
        current = get_nested_value(data, path)
        if current is None or current == "":
            set_nested_value(data, path, value_to_set)
            filled += 1
    return filled


def update_questionnaire_from_image(
    data,
    text_pages,
    image_by_page,
    deepseek_state,
    prompt,
    image_format,
    image_quality,
    debug,
    checkbox_image_by_page=None,
    questionnaire_pages=None,
    checkbox_deskew=False,
    checkbox_contrast=1.0,
    checkbox_binarize=True,
    checkbox_crop_left=0.0,
    checkbox_debug_dir=None,
    fallback_to_ocr=False,
):
    if questionnaire_pages is None:
        questionnaire_pages = find_questionnaire_pages(text_pages)
    if not questionnaire_pages:
        return
    questionnaire_pages = sorted(questionnaire_pages)
    source_images = checkbox_image_by_page or image_by_page

    checkbox_rows = []
    opencv_missing = False
    for page_num in questionnaire_pages:
        image = source_images.get(page_num)
        if image is None:
            continue
        rows = detect_checkbox_rows(
            image,
            deskew=checkbox_deskew,
            contrast=checkbox_contrast,
            binarize=checkbox_binarize,
            crop_left=checkbox_crop_left,
            debug_dir=checkbox_debug_dir,
            page_num=page_num,
        )
        if rows is None:
            opencv_missing = True
            break
        log(f"Detected {len(rows)} checkbox rows on page {page_num}.")
        checkbox_rows.extend(rows)

    if checkbox_rows:
        expected = len(QUESTIONNAIRE_FIELDS)
        if len(checkbox_rows) != expected:
            log(
                f"Checkbox row count {len(checkbox_rows)} "
                f"(expected {expected}). Mapping in fixed order."
            )
        filled = apply_checkbox_rows(data, checkbox_rows)
        log(
            f"Checkbox detection filled {filled} fields "
            f"from {len(checkbox_rows)} rows."
        )
        return

    if opencv_missing:
        log(
            "OpenCV not installed; falling back to DeepSeek questionnaire OCR. "
            "Install with: python -m pip install opencv-python"
        )
        if not fallback_to_ocr:
            return
    else:
        log("No checkbox rows detected.")
        if not fallback_to_ocr:
            return
        log("Falling back to DeepSeek questionnaire OCR.")

    for page_num in questionnaire_pages:
        image = source_images.get(page_num)
        if image is None:
            continue
        log(f"DeepSeek questionnaire page {page_num} start")
        call_start = time.perf_counter()
        response = deepseek_ocr_image(
            deepseek_state,
            prompt,
            image,
            image_format,
            image_quality,
            page_num,
        )
        log(
            f"DeepSeek questionnaire page {page_num} done in "
            f"{time.perf_counter() - call_start:.1f}s"
        )
        if debug:
            debug_path = f"deepseek_questionnaire_page_{page_num}.txt"
            with open(debug_path, "w", encoding="utf-8") as f:
                f.write(response)
        payload = parse_questionnaire_response(response)
        if not payload:
            continue
        for key, path, _ in QUESTIONNAIRE_FIELDS:
            raw_value = None
            if key in payload:
                raw_value = payload.get(key)
            elif path in payload:
                raw_value = payload.get(path)
            if raw_value is None:
                continue
            normalized = normalize_yes_no_value(raw_value)
            if normalized is None and isinstance(raw_value, str) and raw_value.strip() == "":
                continue
            if path.endswith(".exporta") and isinstance(normalized, bool):
                value_to_set = "si" if normalized else "no"
            else:
                value_to_set = normalized
            if value_to_set is None:
                continue
            current = get_nested_value(data, path)
            if current is None or current == "":
                set_nested_value(data, path, value_to_set)
