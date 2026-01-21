import json
import os


def load_schema(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def enforce_schema(schema, data):
    if schema is None:
        return data

    if isinstance(schema, dict):
        data = data if isinstance(data, dict) else {}
        result = {}
        for key, value in schema.items():
            result[key] = enforce_schema(value, data.get(key))
        return result

    if isinstance(schema, list):
        item_schema = schema[0] if schema else None
        if not isinstance(data, list):
            return []
        if item_schema is None:
            return data
        return [enforce_schema(item_schema, item) for item in data]

    if data is None:
        return schema

    if isinstance(schema, bool):
        return data if isinstance(data, bool) else schema
    if isinstance(schema, int) and not isinstance(schema, bool):
        return data if isinstance(data, int) and not isinstance(data, bool) else schema
    if isinstance(schema, float):
        if isinstance(data, (int, float)) and not isinstance(data, bool):
            return float(data)
        return schema
    if isinstance(schema, str):
        return data if isinstance(data, str) else schema

    return data


def flatten_json(value, prefix="", out=None):
    if out is None:
        out = {}
    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else key
            flatten_json(child, child_prefix, out)
    elif isinstance(value, list):
        if not value:
            out[prefix] = ""
        else:
            for idx, item in enumerate(value, start=1):
                child_prefix = f"{prefix}[{idx}]"
                flatten_json(item, child_prefix, out)
    else:
        out[prefix] = value
    return out


def flatten_for_excel(data, columns):
    row = flatten_json(data)
    ordered = {key: row.get(key, "") for key in columns}
    return ordered


def resolve_output_json_path(out_arg, pdf_path, multiple):
    if not out_arg:
        return os.path.splitext(pdf_path)[0] + ".json"
    if not multiple:
        return out_arg
    if out_arg.lower().endswith(".json"):
        raise SystemExit(
            "--out must be a directory when processing multiple PDFs."
        )
    os.makedirs(out_arg, exist_ok=True)
    base = os.path.splitext(os.path.basename(pdf_path))[0] + ".json"
    return os.path.join(out_arg, base)
