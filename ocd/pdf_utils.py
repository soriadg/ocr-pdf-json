import os


def get_pdf_page_count(pdf_path):
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: pypdf. Install with: python -m pip install pypdf"
        ) from exc
    reader = PdfReader(pdf_path)
    return len(reader.pages)


def group_page_ranges(page_numbers):
    if not page_numbers:
        return []
    ranges = []
    start = page_numbers[0]
    end = start
    for page in page_numbers[1:]:
        if page == end + 1:
            end = page
        else:
            ranges.append((start, end))
            start = page
            end = page
    ranges.append((start, end))
    return ranges


def render_pdf_images(
    pdf_path,
    dpi,
    page_numbers=None,
    thread_count=None,
    use_pdftocairo=None,
):
    try:
        from pdf2image import convert_from_path
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: pdf2image. Install with: python -m pip install pdf2image pillow"
        ) from exc

    if page_numbers:
        pages = list(page_numbers)
    else:
        pages = None

    if thread_count is None:
        thread_count = default_thread_count()

    options = {
        "dpi": dpi,
        "thread_count": thread_count,
    }
    if use_pdftocairo is not None:
        options["use_pdftocairo"] = use_pdftocairo

    if pages is None:
        return convert_from_path(pdf_path, **options)

    images = []
    for start, end in group_page_ranges(pages):
        images.extend(
            convert_from_path(
                pdf_path,
                first_page=start,
                last_page=end,
                **options,
            )
        )
    return images


def parse_page_range(page_range, total_pages):
    if not page_range or not page_range.strip():
        return list(range(1, total_pages + 1))

    result = set()
    for part in page_range.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_str, end_str = part.split("-", 1)
            try:
                start = int(start_str) if start_str.strip() else 1
                end = int(end_str) if end_str.strip() else total_pages
            except ValueError as exc:
                raise SystemExit(
                    "Invalid --page-range. Use format like 1-3,5,7-."
                ) from exc
            if start > end:
                raise SystemExit(
                    "Invalid --page-range. Use format like 1-3,5,7-."
                )
            start = max(1, start)
            end = min(total_pages, end)
            for page in range(start, end + 1):
                result.add(page)
        else:
            try:
                page = int(part)
            except ValueError as exc:
                raise SystemExit(
                    "Invalid --page-range. Use format like 1-3,5,7-."
                ) from exc
            if 1 <= page <= total_pages:
                result.add(page)

    if not result:
        raise SystemExit("No pages selected by --page-range.")

    return sorted(result)


def format_page_list(pages, limit=20):
    if not pages:
        return "[]"
    if len(pages) <= limit:
        return ",".join(str(page) for page in pages)
    head = ",".join(str(page) for page in pages[:10])
    tail = ",".join(str(page) for page in pages[-10:])
    return f"{head},...,{tail} (count={len(pages)})"


def default_thread_count():
    count = os.cpu_count() or 1
    return max(1, min(count, 4))
