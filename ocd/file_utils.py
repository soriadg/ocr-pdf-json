import os


def find_default_pdf(cwd):
    pdfs = [f for f in os.listdir(cwd) if f.lower().endswith(".pdf")]
    if len(pdfs) == 1:
        return os.path.join(cwd, pdfs[0])
    return None


def list_pdfs(directory, recursive=False):
    try:
        entries = os.listdir(directory)
    except OSError as exc:
        raise SystemExit(f"Cannot read directory: {directory}. {exc}") from exc

    if recursive:
        pdfs = []
        for root, _, files in os.walk(directory):
            for name in files:
                if name.lower().endswith(".pdf"):
                    pdfs.append(os.path.join(root, name))
        return sorted(pdfs)

    pdfs = [
        os.path.join(directory, name)
        for name in entries
        if name.lower().endswith(".pdf")
    ]
    return sorted(pdfs)
