import copy
import os
import re

from .questionnaire import parse_yes_no_from_line
from .text_utils import (
    extract_latest_value,
    extract_list_from_value,
    extract_section_lines,
    extract_table_numeric_rows,
    extract_table_rows,
    find_line_index,
    find_line_with_keywords,
    find_value_in_lines,
    line_contains_any,
    to_number,
)


def combine_text_pages(text_pages):
    if not text_pages:
        return ""
    text_pages.sort(key=lambda item: item[0])
    parts = []
    for page_num, text in text_pages:
        cleaned = text.strip()
        if cleaned:
            parts.append(f"[PAGE {page_num}]\n{cleaned}")
    return "\n\n".join(parts).strip()


def extract_title_from_text(text):
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return ""
    candidates = lines[:8]
    for line in candidates:
        if len(line) < 6:
            continue
        if line.lower().startswith("page "):
            continue
        return line
    return ""


def extract_document_type(title, fallback_name):
    title_lower = title.lower()
    keywords = [
        "diagnostico",
        "informe",
        "reporte",
        "resumen",
        "estudio",
        "plan",
        "manual",
        "memoria",
    ]
    for keyword in keywords:
        if keyword in title_lower:
            return keyword
    name_lower = fallback_name.lower()
    for keyword in keywords:
        if keyword in name_lower:
            return keyword
    return ""


def extract_document_date(text):
    patterns = [
        r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
        r"\b\d{1,2}\s+de\s+(enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|octubre|noviembre|diciembre)\s+de\s+\d{4}\b",
        r"\b20\d{2}\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(0)
    return ""


def extract_resultado(lines, data):
    for entry in data.get("resultado", {}).get("secciones", []):
        name = entry.get("seccion", "")
        if not name:
            continue
        pattern_base = re.escape(name)
        obtained_match = None
        percent_match = None
        for line in lines:
            if re.search(pattern_base, line, flags=re.IGNORECASE):
                obtained_match = re.search(
                    rf"{pattern_base}.*?:\s*(\d+[\d.,]*)",
                    line,
                    flags=re.IGNORECASE,
                )
                percent_match = re.search(
                    rf"{pattern_base}\s*%\s*:?\s*(\d+[\d.,]*)",
                    line,
                    flags=re.IGNORECASE,
                )
                if obtained_match or percent_match:
                    break
        if obtained_match:
            entry["obtenido"] = to_number(obtained_match.group(1))
        if percent_match:
            entry["porcentaje"] = to_number(percent_match.group(1))
    total_entry = data.get("resultado", {}).get("total", {})
    for line in lines:
        if line_contains_any(line, ["Total"]):
            match_total = re.search(r"Total\s*:?\s*(\d+[\d.,]*)", line, flags=re.IGNORECASE)
            match_pct = re.search(r"Total\s*%\s*:?\s*(\d+[\d.,]*)", line, flags=re.IGNORECASE)
            if match_total:
                total_entry["obtenido"] = to_number(match_total.group(1))
            if match_pct:
                total_entry["porcentaje"] = to_number(match_pct.group(1))
            break


def build_json_from_ocr(template, text_pages, pdf_path):
    data = copy.deepcopy(template)
    combined = combine_text_pages(text_pages)
    lines = [line.strip() for line in combined.splitlines() if line.strip()]

    data["fecha"] = extract_document_date(combined) or data.get("fecha")
    data.setdefault("metadata", {})
    data["metadata"]["fuente_archivo"] = os.path.basename(pdf_path)

    empresa = data.get("empresa", {})
    empresa["razon_social"] = find_value_in_lines(
        lines,
        [
            "Empresa o Razon Social",
            "Empresa o RazA3n Social",
            "Razon Social",
            "Razon Social",
            "RazA3n Social",
            "Nombre de Cuenta",
        ],
    ) or empresa.get("razon_social", "")
    empresa["direccion"] = find_value_in_lines(
        lines,
        ["Direccion", "DirecciA3n"],
    ) or empresa.get("direccion", "")
    empresa["ciudad"] = find_value_in_lines(lines, ["Ciudad"]) or empresa.get("ciudad", "")
    empresa["telefono"] = find_value_in_lines(
        lines,
        ["Telefono", "TelAcfono", "TelAfono"],
    ) or empresa.get("telefono", "")
    empresa["celular"] = find_value_in_lines(lines, ["Celular"]) or empresa.get("celular", "")
    empresa["ruc"] = find_value_in_lines(lines, ["Ruc", "RUC"]) or empresa.get("ruc", "")
    empresa["email"] = find_value_in_lines(
        lines,
        ["Correo", "E-mail", "Email", "Correo electrA3nico", "Correo electronico"],
    ) or empresa.get("email", "")
    empresa["sitio_web"] = find_value_in_lines(
        lines,
        ["Sitio Web", "Sito Web", "Sitio Web:", "Website"],
    ) or empresa.get("sitio_web", "")
    data["empresa"] = empresa

    rep_lines = extract_section_lines(
        lines,
        ["Representante ante REDIEX", "Representante ante", "Representante"],
        ["Informacion del Sector", "Informacion del Sector", "Sector", "D - Informacion del Sector"],
    )
    representante = data.get("representante_rediex", {})
    representante["nombre"] = find_value_in_lines(rep_lines, ["Nombre"]) or representante.get("nombre", "")
    if not representante.get("nombre"):
        representante["nombre"] = find_value_in_lines(
            rep_lines, ["Representante ante REDIEX", "Representante ante"]
        )
    representante["cargo"] = find_value_in_lines(rep_lines, ["Cargo"]) or representante.get("cargo", "")
    representante["telefono"] = find_value_in_lines(
        rep_lines,
        ["Telefono", "Telefono1", "Telefono 1"],
    ) or representante.get("telefono", "")
    representante["celular"] = find_value_in_lines(
        rep_lines,
        ["Celular", "Celular1", "Celular 1"],
    ) or representante.get("celular", "")
    representante["email"] = find_value_in_lines(rep_lines, ["E-mail", "Email", "Correo"]) or representante.get(
        "email", ""
    )
    representante["ci"] = find_value_in_lines(rep_lines, ["CI", "C.I"]) or representante.get("ci", "")
    data["representante_rediex"] = representante

    sector_lines = extract_section_lines(
        lines,
        ["Informacion del Sector", "Informacion del Sector", "Sector"],
        ["Facturacion", "Facturacion", "FacturaciA3n"],
    )
    clasificacion = data.get("clasificacion", {})
    clasificacion["sector"] = find_value_in_lines(sector_lines, ["Sector"]) or clasificacion.get("sector", "")
    clasificacion["sub_sector"] = find_value_in_lines(
        sector_lines,
        ["Sub-Sector", "Sub Sector", "Sub-Sector", "Industria"],
    ) or clasificacion.get("sub_sector", "")
    clasificacion["rubro_especifico"] = find_value_in_lines(
        sector_lines,
        ["Rubro Especifico", "Rubro EspecA-fico", "Rubro EspecAfico", "Alimentos elaborados"],
    ) or clasificacion.get("rubro_especifico", "")
    clasificacion["otros"] = find_value_in_lines(
        sector_lines,
        ["Descripcion", "Descripcion Rubro", "Descripcion Sub-Sector"],
    ) or clasificacion.get("otros", "")
    data["clasificacion"] = clasificacion

    facturacion = data.get("facturacion_usd", {})
    facturacion["ventas_locales"] = to_number(extract_latest_value(lines, ["Ventas Locales"])) or facturacion.get(
        "ventas_locales"
    )
    facturacion["exportaciones"] = to_number(extract_latest_value(lines, ["Exportaciones"])) or facturacion.get(
        "exportaciones"
    )
    if facturacion.get("ventas_locales") is None or facturacion.get("exportaciones") is None:
        fact_idx = find_line_index(lines, ["Facturacion", "Facturacion", "FacturaciA3n"])
        fact_rows = extract_table_numeric_rows(lines, fact_idx, max_rows=3)
        if fact_rows:
            if facturacion.get("ventas_locales") is None and len(fact_rows) >= 1:
                facturacion["ventas_locales"] = to_number(fact_rows[0][-1])
            if facturacion.get("exportaciones") is None and len(fact_rows) >= 2:
                facturacion["exportaciones"] = to_number(fact_rows[1][-1])
    data["facturacion_usd"] = facturacion

    empleo = data.get("empleo", {})
    empleo["cantidad_empleados"] = to_number(extract_latest_value(lines, ["Empleados"])) or empleo.get(
        "cantidad_empleados"
    )
    empleo["cantidad_mujeres_empleadas"] = to_number(
        extract_latest_value(lines, ["Mujeres empleadas"])
    ) or empleo.get("cantidad_mujeres_empleadas")
    empleo["cantidad_mujeres_puestos_gerenciales"] = to_number(
        extract_latest_value(lines, ["Mujeres en Puestos Gerenciales", "Mujeres en Puestos Gerenciales"])
    ) or empleo.get("cantidad_mujeres_puestos_gerenciales")
    empleo["empleo_indirecto"] = to_number(extract_latest_value(lines, ["Empleo Indirecto"])) or empleo.get(
        "empleo_indirecto"
    )
    empleo["porcentaje_mujeres_sobre_total"] = to_number(
        extract_latest_value(lines, ["% Mujeres", "Porcentaje Mujeres"])
    ) or empleo.get("porcentaje_mujeres_sobre_total")
    if (
        empleo.get("cantidad_empleados") is None
        or empleo.get("cantidad_mujeres_empleadas") is None
        or empleo.get("cantidad_mujeres_puestos_gerenciales") is None
        or empleo.get("porcentaje_mujeres_sobre_total") is None
    ):
        empleo_idx = find_line_index(
            lines,
            ["Empleo", "Empleados", "% Exp s/ Fact", "% Exp s/ Fact. Total"],
        )
        empleo_rows = extract_table_numeric_rows(lines, empleo_idx, max_rows=4)
        if empleo_rows:
            if empleo.get("cantidad_empleados") is None and len(empleo_rows) >= 1:
                empleo["cantidad_empleados"] = to_number(empleo_rows[0][-1])
            if empleo.get("cantidad_mujeres_empleadas") is None and len(empleo_rows) >= 2:
                empleo["cantidad_mujeres_empleadas"] = to_number(empleo_rows[1][-1])
            if empleo.get("cantidad_mujeres_puestos_gerenciales") is None and len(empleo_rows) >= 3:
                empleo["cantidad_mujeres_puestos_gerenciales"] = to_number(empleo_rows[2][-1])
            if empleo.get("porcentaje_mujeres_sobre_total") is None and len(empleo_rows) >= 4:
                empleo["porcentaje_mujeres_sobre_total"] = to_number(empleo_rows[3][-1])
    data["empleo"] = empleo

    ubicaciones = data.get("ubicaciones_dependencias", [])
    for idx in range(1, 6):
        labels = {
            "tipo": [f"Tipo {idx}", f"Tipo {idx}A", f"Tipo {idx}\u00ba"],
            "municipio": [f"Municipio {idx}", f"Municipio {idx}A", f"Municipio {idx}\u00ba"],
            "departamento": [f"Departamento {idx}", f"Departamento {idx}A", f"Departamento {idx}\u00ba"],
            "empleos": [f"Empleos {idx}", f"Empleos {idx}A", f"Empleos {idx}\u00ba"],
        }
        if idx - 1 < len(ubicaciones):
            ubicaciones[idx - 1]["tipo"] = find_value_in_lines(lines, labels["tipo"]) or ubicaciones[idx - 1].get(
                "tipo", ""
            )
            ubicaciones[idx - 1]["municipio"] = find_value_in_lines(
                lines, labels["municipio"]
            ) or ubicaciones[idx - 1].get("municipio", "")
            ubicaciones[idx - 1]["departamento"] = find_value_in_lines(
                lines, labels["departamento"]
            ) or ubicaciones[idx - 1].get("departamento", "")
            empleos_value = find_value_in_lines(lines, labels["empleos"])
            if empleos_value:
                ubicaciones[idx - 1]["empleos"] = to_number(empleos_value)
    if ubicaciones and not any(item.get("tipo") for item in ubicaciones):
        ubic_idx = find_line_index(
            lines,
            [
                "Tipo | Municipio | Departamento | Empleos",
                "Ubicaciones de las dependencias",
            ],
        )
        table_rows = extract_table_rows(
            lines,
            ubic_idx,
            min_columns=4,
            max_rows=10,
            stop_keywords=["Omitir", "Cuestionario"],
        )
        parsed = []
        for row in table_rows:
            row_text = " ".join(row)
            if line_contains_any(row_text, ["Tipo", "Municipio", "Departamento", "Empleos"]):
                continue
            if not re.search(r"[A-Za-z]", row_text):
                continue
            parsed.append(row)
        for idx, row in enumerate(parsed[:5]):
            if idx >= len(ubicaciones):
                break
            ubicaciones[idx]["tipo"] = row[0] if len(row) > 0 else ""
            ubicaciones[idx]["municipio"] = row[1] if len(row) > 1 else ""
            ubicaciones[idx]["departamento"] = row[2] if len(row) > 2 else ""
            if len(row) > 3:
                ubicaciones[idx]["empleos"] = to_number(row[3])
    data["ubicaciones_dependencias"] = ubicaciones

    cuestionario_lines = extract_section_lines(
        lines,
        ["Cuestionario", "Cuestionario 1", "Cuestionario 1."],
        ["RESULTADO", "Resultado"],
    )
    cuestionario = data.get("cuestionario", {})
    recursos = cuestionario.get("1_recursos", {})
    recursos["a_pagina_web"] = parse_yes_no_from_line(
        find_line_with_keywords(cuestionario_lines, ["Pagina web"])
    )
    recursos["b_cuenta_bancaria"] = parse_yes_no_from_line(
        find_line_with_keywords(cuestionario_lines, ["Cuenta bancaria"])
    )
    recursos["c_encargado_comercio_exterior"] = parse_yes_no_from_line(
        find_line_with_keywords(cuestionario_lines, ["persona encargada", "comercio"])
    )
    recursos["d_personal_ingles_idioma_objetivo"] = parse_yes_no_from_line(
        find_line_with_keywords(cuestionario_lines, ["idioma", "mercado objetivo"])
    )
    recursos["e_capital_para_demanda_exterior"] = parse_yes_no_from_line(
        find_line_with_keywords(cuestionario_lines, ["capital", "demanda"])
    )
    recursos["f_material_promocional"] = parse_yes_no_from_line(
        find_line_with_keywords(cuestionario_lines, ["material promocional"])
    )
    cuestionario["1_recursos"] = recursos

    experiencia = cuestionario.get("2_experiencia", {})
    experiencia["a_antiguedad_mercado_local"] = find_value_in_lines(
        cuestionario_lines,
        ["Hace cuantos anos", "Hace cuantos"],
    )
    experiencia["b_mas_de_3_ciudades_importantes_py"] = parse_yes_no_from_line(
        find_line_with_keywords(cuestionario_lines, ["ciudades", "importantes", "paraguay"])
    )
    exporta_line = find_line_with_keywords(
        cuestionario_lines,
        ["Comercializa la empresa", "empresa exporta", "exporta la empresa"],
    )
    exporta_value = parse_yes_no_from_line(exporta_line)
    exporta = experiencia.get("d_empresa_exporta", {})
    exporta["exporta"] = "si" if exporta_value is True else ("no" if exporta_value is False else "")
    exporta["experiencia_exportadora"] = find_value_in_lines(
        cuestionario_lines, ["Experiencia exportadora"]
    )
    exporta["antiguedad_exportadora"] = find_value_in_lines(
        cuestionario_lines, ["Antiguedad exportadora", "AntigA\u00acedad exportadora", "AntigAA\u00aaedad exportadora"]
    )
    exporta["a_cuantos_paises_exporta"] = to_number(
        find_value_in_lines(cuestionario_lines, ["A cuantos paises exporta", "A cuantos paA-ses exporta"])
    )
    exporta["a_cuales_paises_exporta"] = extract_list_from_value(
        find_value_in_lines(cuestionario_lines, ["A cuales paises exporta", "A cuA-les paA-ses exporta"])
    )
    experiencia["d_empresa_exporta"] = exporta
    experiencia["g_cuantos_productos_distintos"] = to_number(
        find_value_in_lines(cuestionario_lines, ["Cuantos productos distintos exporta"])
    )
    experiencia["h_que_productos_exporta"] = extract_list_from_value(
        find_value_in_lines(cuestionario_lines, ["Que productos exporta", "QuAc productos exporta"])
    )
    cuestionario["2_experiencia"] = experiencia

    producto = cuestionario.get("3_producto", {})
    producto["a_producto_servicio_propio_comercializado"] = parse_yes_no_from_line(
        find_line_with_keywords(cuestionario_lines, ["Producto/Servicio propio", "Producto propio"])
    )
    producto["b_capacidad_aumentar_produccion_ventas_internacionales"] = parse_yes_no_from_line(
        find_line_with_keywords(cuestionario_lines, ["capacidad", "aumentar"])
    )
    producto["c_certificacion_internacional"] = parse_yes_no_from_line(
        find_line_with_keywords(cuestionario_lines, ["certificacion internacional", "certificaciA3n internacional"])
    )
    producto["c_certificacion_detalle"] = find_value_in_lines(
        cuestionario_lines,
        ["certificacion", "certificaciA3n"],
    )
    cuestionario["3_producto"] = producto

    planificacion = cuestionario.get("4_planificacion", {})
    planificacion["a_presupuesto_anual"] = parse_yes_no_from_line(
        find_line_with_keywords(cuestionario_lines, ["Presupuesto Anual", "Presupuesto anual"])
    )
    planificacion["b_objetivos_metas_estrategias_mercado_local"] = parse_yes_no_from_line(
        find_line_with_keywords(cuestionario_lines, ["objetivos", "metas", "estrategias"])
    )
    planificacion["c_plan_exportacion"] = parse_yes_no_from_line(
        find_line_with_keywords(cuestionario_lines, ["Plan de exportacion", "Plan de exportaciA3n"])
    )
    cuestionario["4_planificacion"] = planificacion

    conocimiento = cuestionario.get("5_conocimiento_mercado_exterior", {})
    normas = conocimiento.get("a_conoce_normas_y_entorno", {})
    normas["normas_envase"] = parse_yes_no_from_line(
        find_line_with_keywords(cuestionario_lines, ["envase"])
    )
    normas["normas_embalaje"] = parse_yes_no_from_line(
        find_line_with_keywords(cuestionario_lines, ["embalaje"])
    )
    normas["normas_etiquetado"] = parse_yes_no_from_line(
        find_line_with_keywords(cuestionario_lines, ["etiquetado"])
    )
    normas["estandares_calidad_y_certificaciones"] = parse_yes_no_from_line(
        find_line_with_keywords(cuestionario_lines, ["estandares", "calidad"])
    )
    normas["aspectos_culturales"] = parse_yes_no_from_line(
        find_line_with_keywords(cuestionario_lines, ["aspectos culturales", "culturales"])
    )
    normas["competidores"] = parse_yes_no_from_line(
        find_line_with_keywords(cuestionario_lines, ["competidores"])
    )
    conocimiento["a_conoce_normas_y_entorno"] = normas
    conocimiento["b_conoce_medios_transporte_y_costos"] = parse_yes_no_from_line(
        find_line_with_keywords(cuestionario_lines, ["medios", "transporte"])
    )
    conocimiento["c_conoce_canales_comercializacion"] = parse_yes_no_from_line(
        find_line_with_keywords(cuestionario_lines, ["canales", "comercializacion"])
    )
    cuestionario["5_conocimiento_mercado_exterior"] = conocimiento

    barreras = cuestionario.get("6_barreras", {})
    barreras["a_conoce_barreras_arancelarias_no_arancelarias"] = parse_yes_no_from_line(
        find_line_with_keywords(cuestionario_lines, ["barreras", "arancelarias"])
    )
    barreras["b_conoce_acuerdos_que_benefician_producto"] = parse_yes_no_from_line(
        find_line_with_keywords(cuestionario_lines, ["acuerdos", "paraguay"])
    )
    barreras["c_conoce_regulaciones_internacionales"] = parse_yes_no_from_line(
        find_line_with_keywords(cuestionario_lines, ["regulaciones internacionales"])
    )
    cuestionario["6_barreras"] = barreras

    precios = cuestionario.get("7_precios", {})
    precios["a_conoce_precios_mayoristas_minoristas_competencia"] = parse_yes_no_from_line(
        find_line_with_keywords(cuestionario_lines, ["precios", "mayoristas"])
    )
    precios["b_conoce_estructura_margenes_industria"] = parse_yes_no_from_line(
        find_line_with_keywords(cuestionario_lines, ["estructura", "margenes"])
    )
    cuestionario["7_precios"] = precios

    ambiental = cuestionario.get("8_desempeno_ambiental", {})
    ambiental["a_tiene_habilitacion_ambiental"] = parse_yes_no_from_line(
        find_line_with_keywords(cuestionario_lines, ["habilitacion ambiental"])
    )
    ambiental["observaciones"] = find_value_in_lines(cuestionario_lines, ["Observaciones"])
    cuestionario["8_desempeno_ambiental"] = ambiental

    data["cuestionario"] = cuestionario

    extract_resultado(lines, data)
    return data
