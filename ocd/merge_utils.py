def merge_missing(primary, fallback):
    if isinstance(primary, dict) and isinstance(fallback, dict):
        merged = {}
        for key in primary.keys() | fallback.keys():
            if key in primary and key in fallback:
                merged[key] = merge_missing(primary[key], fallback[key])
            elif key in primary:
                merged[key] = primary[key]
            else:
                merged[key] = fallback[key]
        return merged

    if isinstance(primary, list):
        return fallback if not primary else primary

    if primary is None or primary == "":
        return fallback if fallback not in (None, "") else primary

    return primary
