import re
import unicodedata


def normalize_for_match(text):
    if text is None:
        return ""
    text = str(text).lower()
    text = text.replace("a3", "o")
    text = text.replace("aA\u00f1", "n")
    text = text.replace("aA\u00ad", "i")
    text = text.replace("aAc", "e")
    text = text.replace("aA\u00a7", "o")
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return text.strip()


def line_contains_any(line, keywords):
    hay = normalize_for_match(line)
    for keyword in keywords:
        if normalize_for_match(keyword) in hay:
            return True
    return False


def clean_cell_value(value):
    if value is None:
        return ""
    cleaned = str(value).strip()
    cleaned = cleaned.strip("|").strip()
    if not cleaned:
        return ""
    if re.fullmatch(r"-+", cleaned):
        return ""
    return cleaned


def extract_value_from_line(line, labels):
    if not line_contains_any(line, labels):
        return None
    if "|" in line:
        parts = [part.strip() for part in line.split("|") if part.strip()]
        for idx, part in enumerate(parts):
            if line_contains_any(part, labels):
                if idx + 1 < len(parts):
                    value = clean_cell_value(parts[idx + 1])
                    return value or None
    if ":" in line:
        left, right = line.split(":", 1)
        if line_contains_any(left, labels):
            value = clean_cell_value(right)
            return value or None
    return None


def find_value_in_lines(lines, labels):
    for line in lines:
        value = extract_value_from_line(line, labels)
        if value:
            return value
    return ""


def find_line_index(lines, keywords):
    for idx, line in enumerate(lines):
        if line_contains_any(line, keywords):
            return idx
    return None


def extract_table_rows(lines, start_idx, min_columns=2, max_rows=None, stop_keywords=None):
    if start_idx is None:
        return []
    rows = []
    for line in lines[start_idx + 1 :]:
        if stop_keywords and line_contains_any(line, stop_keywords):
            break
        if not line:
            if rows:
                break
            continue
        if "|" not in line:
            if rows:
                break
            continue
        cells = [cell.strip() for cell in line.split("|") if cell.strip()]
        if not cells:
            if rows:
                break
            continue
        if all(re.fullmatch(r"-+", cell) for cell in cells):
            continue
        if len(cells) < min_columns:
            continue
        rows.append(cells)
        if max_rows and len(rows) >= max_rows:
            break
    return rows


def extract_table_numeric_rows(lines, start_idx, max_rows=None):
    rows = []
    for cells in extract_table_rows(lines, start_idx, min_columns=2, max_rows=None):
        if all(re.fullmatch(r"\d{4}", cell) for cell in cells):
            continue
        numeric_cells = [cell for cell in cells if re.fullmatch(r"[-%\d.,]+", cell)]
        if len(numeric_cells) < 2:
            if rows:
                break
            continue
        rows.append(numeric_cells)
        if max_rows and len(rows) >= max_rows:
            break
    return rows


def to_number(value):
    if value is None:
        return None
    cleaned = str(value)
    cleaned = cleaned.replace("US$", "").replace("$", "")
    cleaned = cleaned.replace("%", "")
    cleaned = cleaned.replace(" ", "")
    cleaned = cleaned.replace(".", "")
    cleaned = cleaned.replace(",", ".")
    cleaned = cleaned.strip()
    if not cleaned:
        return None
    try:
        number = float(cleaned)
    except ValueError:
        return None
    if number.is_integer():
        return int(number)
    return number


def extract_section_lines(lines, start_keywords, end_keywords):
    start_idx = None
    for idx, line in enumerate(lines):
        if line_contains_any(line, start_keywords):
            start_idx = idx
            break
    if start_idx is None:
        return []
    end_idx = len(lines)
    for idx in range(start_idx + 1, len(lines)):
        if line_contains_any(lines[idx], end_keywords):
            end_idx = idx
            break
    return lines[start_idx:end_idx]


def find_line_with_keywords(lines, keywords):
    for line in lines:
        if line_contains_any(line, keywords):
            return line
    return ""


def extract_latest_value(lines, labels):
    candidates = []
    for line in lines:
        if line_contains_any(line, labels):
            value = extract_value_from_line(line, labels)
            if not value:
                continue
            year_match = re.search(r"(20\d{2})", line)
            year = int(year_match.group(1)) if year_match else -1
            candidates.append((year, value))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[-1][1]


def extract_list_from_value(value):
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]
