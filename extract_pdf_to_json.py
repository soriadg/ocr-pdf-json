import argparse
import atexit
import json
import os
import shutil
import time

from ocd.deepseek_ocr import (
    extract_text_pages_with_deepseek,
    find_local_deepseek_model,
    init_deepseek_ocr_state,
    normalize_deepseek_prompt,
    release_deepseek_state,
)
from ocd.excel_utils import open_excel_file, write_excel
from ocd.file_utils import find_default_pdf, list_pdfs
from ocd.json_builder import build_json_from_ocr
from ocd.json_llm import (
    SYSTEM_PROMPT as JSON_LLM_SYSTEM_PROMPT,
    build_user_prompt as build_json_llm_user_prompt,
    extract_json_from_text,
    generate_json_with_llm,
    load_json_llm,
)
from ocd.logging_utils import close_logger, log, setup_logger
from ocd.merge_utils import merge_missing
from ocd.pdf_utils import (
    default_thread_count,
    format_page_list,
    get_pdf_page_count,
    parse_page_range,
    render_pdf_images,
)
from ocd.questionnaire import (
    build_questionnaire_prompt,
    find_questionnaire_pages,
    update_questionnaire_from_image,
)
from ocd.schema_utils import (
    flatten_for_excel,
    flatten_json,
    load_schema,
    resolve_output_json_path,
)


def main():
    parser = argparse.ArgumentParser(
        description="Extract JSON from PDFs using the DeepSeek OCR Hugging Face model."
    )
    parser.add_argument("--pdf", help="Path to a PDF file or a directory of PDFs.")
    parser.add_argument(
        "--input-dir",
        help="Directory of PDFs to process.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        default=False,
        help="Search input directories recursively for PDFs.",
    )
    parser.add_argument(
        "--schema",
        help="Path to JSON template. Defaults to schema.json next to this script.",
    )
    parser.add_argument(
        "--base-json",
        help="Path to base JSON template (e.g. 'JSON base.txt').",
    )
    parser.add_argument(
        "--mode",
        choices=["auto", "vision", "text"],
        default="auto",
        help="Vision-only JSON extraction (auto/text kept for compatibility).",
    )
    parser.add_argument(
        "--model",
        help="Override the DeepSeek OCR model path or Hugging Face ID.",
    )
    parser.add_argument(
        "--vision-model",
        default="deepseek-ai/DeepSeek-OCR",
        help="Hugging Face model ID or local path for OCR.",
    )
    parser.add_argument(
        "--deepseek-prompt",
        default="<image>\nFree OCR.",
        help="Prompt used for OCR text extraction.",
    )
    parser.add_argument(
        "--deepseek-base-size",
        type=int,
        default=1024,
        help="DeepSeek OCR base image size.",
    )
    parser.add_argument(
        "--deepseek-image-size",
        type=int,
        default=640,
        help="DeepSeek OCR image size.",
    )
    parser.add_argument(
        "--deepseek-max-new-tokens",
        type=int,
        default=2048,
        help="Max tokens to generate per OCR call (0 to disable clamp).",
    )
    parser.add_argument(
        "--deepseek-crop-mode",
        action="store_true",
        default=True,
        help="Enable DeepSeek OCR crop mode.",
    )
    parser.add_argument(
        "--no-deepseek-crop-mode",
        action="store_false",
        dest="deepseek_crop_mode",
        help="Disable DeepSeek OCR crop mode.",
    )
    parser.add_argument(
        "--deepseek-output-dir",
        help="Directory for DeepSeek OCR temporary outputs.",
    )
    parser.add_argument(
        "--deepseek-cache-dir",
        help="Cache directory for Hugging Face downloads.",
    )
    parser.add_argument(
        "--deepseek-attn-impl",
        help="Attention implementation for DeepSeek OCR (e.g. flash_attention_2).",
    )
    parser.add_argument(
        "--log",
        action="store_true",
        default=True,
        help="Enable progress logging.",
    )
    parser.add_argument(
        "--no-log",
        action="store_false",
        dest="log",
        help="Disable progress logging.",
    )
    parser.add_argument(
        "--log-file",
        help="Write logs to a file instead of stderr.",
    )
    parser.add_argument(
        "--render-threads",
        type=int,
        help="Threads for PDF rendering (default: min(4, CPU)).",
    )
    parser.add_argument(
        "--use-pdftocairo",
        action="store_true",
        default=True,
        help="Use pdftocairo renderer when available.",
    )
    parser.add_argument(
        "--no-use-pdftocairo",
        action="store_false",
        dest="use_pdftocairo",
        help="Disable pdftocairo renderer.",
    )
    parser.add_argument(
        "--image-format",
        choices=["jpeg", "png"],
        default="jpeg",
        help="Image format for OCR input images.",
    )
    parser.add_argument(
        "--image-quality",
        type=int,
        default=70,
        help="JPEG quality (1-95) for OCR input images.",
    )
    parser.add_argument(
        "--out",
        help="Output JSON path (or directory when processing multiple PDFs).",
    )
    parser.add_argument(
        "--out-excel",
        help="Output Excel (.xlsx) path when processing multiple PDFs.",
    )
    parser.add_argument(
        "--ocr-output-dir",
        help="Directory to write OCR text outputs (one .txt per PDF).",
    )
    parser.add_argument(
        "--open-excel",
        action="store_true",
        default=True,
        help="Open the Excel output after writing.",
    )
    parser.add_argument(
        "--no-open-excel",
        action="store_false",
        dest="open_excel",
        help="Do not open the Excel output automatically.",
    )
    parser.add_argument(
        "--questionnaire-from-image",
        action="store_true",
        default=True,
        help="Detect questionnaire checkboxes from images.",
    )
    parser.add_argument(
        "--no-questionnaire-from-image",
        action="store_false",
        dest="questionnaire_from_image",
        help="Disable questionnaire checkbox extraction.",
    )
    parser.add_argument(
        "--questionnaire-ocr-fallback",
        action="store_true",
        default=False,
        help="Fall back to DeepSeek questionnaire OCR if checkbox detection fails.",
    )
    parser.add_argument(
        "--questionnaire-prompt",
        help="Override prompt for questionnaire checkbox extraction.",
    )
    parser.add_argument(
        "--json-llm",
        action="store_true",
        default=False,
        help="Use a text LLM to build JSON from OCR output.",
    )
    parser.add_argument(
        "--json-llm-model",
        help="Hugging Face model ID or local path for JSON LLM.",
    )
    parser.add_argument(
        "--json-llm-cache-dir",
        help="Cache directory for JSON LLM downloads.",
    )
    parser.add_argument(
        "--json-llm-max-new-tokens",
        type=int,
        default=2048,
        help="Max tokens to generate for JSON LLM.",
    )
    parser.add_argument(
        "--json-llm-temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for JSON LLM.",
    )
    parser.add_argument(
        "--json-llm-max-input-chars",
        type=int,
        default=80000,
        help="Max OCR characters passed to JSON LLM (0 = no limit).",
    )
    parser.add_argument(
        "--json-llm-attn-impl",
        help="Attention implementation for JSON LLM (e.g. flash_attention_2).",
    )
    parser.add_argument(
        "--json-llm-debug-dir",
        help="Directory to write raw JSON LLM outputs.",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=3,
        help="Limit number of PDFs processed (0 = no limit).",
    )
    parser.add_argument(
        "--page-range",
        help="Pages to process, e.g. 1-3,5,7- (1-based).",
    )
    parser.add_argument("--ocr-dpi", type=int, default=200)
    parser.add_argument(
        "--checkbox-dpi",
        type=int,
        default=0,
        help="Render questionnaire pages at this DPI for checkbox detection (0 uses OCR DPI).",
    )
    parser.add_argument(
        "--checkbox-deskew",
        action="store_true",
        default=True,
        help="Deskew questionnaire images before checkbox detection.",
    )
    parser.add_argument(
        "--no-checkbox-deskew",
        action="store_false",
        dest="checkbox_deskew",
        help="Disable deskew for questionnaire checkbox detection.",
    )
    parser.add_argument(
        "--checkbox-contrast",
        type=float,
        default=1.6,
        help="Contrast multiplier for questionnaire checkbox detection.",
    )
    parser.add_argument(
        "--checkbox-binarize",
        action="store_true",
        default=True,
        help="Binarize questionnaire images before checkbox detection.",
    )
    parser.add_argument(
        "--no-checkbox-binarize",
        action="store_false",
        dest="checkbox_binarize",
        help="Disable binarization for questionnaire checkbox detection.",
    )
    parser.add_argument(
        "--checkbox-crop-left",
        type=float,
        default=0.0,
        help="Crop this fraction from the left side before checkbox detection (0-0.9).",
    )
    parser.add_argument(
        "--checkbox-debug-dir",
        help="Directory to write checkbox debug images.",
    )
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()

    setup_logger(args.log, args.log_file)
    atexit.register(close_logger)
    if args.log_file:
        log(f"Logging to {args.log_file}")

    if args.render_threads is None:
        args.render_threads = default_thread_count()

    if args.use_pdftocairo and not shutil.which("pdftocairo"):
        log("pdftocairo not found; falling back to pdftoppm.")
        args.use_pdftocairo = False

    if args.vision_model == "deepseek-ai/DeepSeek-OCR":
        local_model = find_local_deepseek_model()
        if local_model:
            args.vision_model = local_model
            log(f"Using local DeepSeek OCR model: {args.vision_model}")

    start_time = time.perf_counter()
    log("Starting extraction")
    log(f"Mode: {args.mode}")
    log(f"DeepSeek model: {args.model or args.vision_model}")
    log(f"OCR dpi: {args.ocr_dpi}")
    if args.checkbox_dpi:
        log(f"Checkbox dpi: {args.checkbox_dpi}")
    if args.questionnaire_from_image:
        log(
            "Checkbox preprocess: "
            f"deskew={args.checkbox_deskew}, "
            f"contrast={args.checkbox_contrast}, "
            f"binarize={args.checkbox_binarize}, "
            f"crop_left={args.checkbox_crop_left}"
        )
        if args.checkbox_debug_dir:
            log(f"Checkbox debug dir: {args.checkbox_debug_dir}")
    if args.ocr_output_dir:
        log(f"OCR output dir: {args.ocr_output_dir}")
    log(f"Render threads: {args.render_threads}")
    log(f"Use pdftocairo: {args.use_pdftocairo}")
    log(f"Image format: {args.image_format}")
    if args.image_format == "jpeg":
        log(f"Image quality: {args.image_quality}")
    log(f"DeepSeek base size: {args.deepseek_base_size}")
    log(f"DeepSeek image size: {args.deepseek_image_size}")
    log(f"DeepSeek max new tokens: {args.deepseek_max_new_tokens}")
    log(f"DeepSeek crop mode: {args.deepseek_crop_mode}")
    log(f"DeepSeek prompt: {normalize_deepseek_prompt(args.deepseek_prompt)!r}")
    if args.questionnaire_from_image:
        log("Questionnaire image pass: enabled")
    if args.json_llm:
        log(f"JSON LLM model: {args.json_llm_model}")
        log(f"JSON LLM max new tokens: {args.json_llm_max_new_tokens}")
        log(f"JSON LLM temperature: {args.json_llm_temperature}")
        log(f"JSON LLM max input chars: {args.json_llm_max_input_chars}")
        if args.json_llm_debug_dir:
            log(f"JSON LLM debug dir: {args.json_llm_debug_dir}")

    pdf_paths = []
    input_dir = args.input_dir
    pdf_arg = args.pdf
    if input_dir:
        if not os.path.isdir(input_dir):
            raise SystemExit(f"Input directory not found: {input_dir}")
        pdf_paths = list_pdfs(input_dir, recursive=args.recursive)
    elif pdf_arg:
        if os.path.isdir(pdf_arg):
            input_dir = pdf_arg
            pdf_paths = list_pdfs(pdf_arg, recursive=args.recursive)
        else:
            pdf_paths = [pdf_arg]
    else:
        default_pdf = find_default_pdf(os.getcwd())
        if default_pdf:
            pdf_paths = [default_pdf]
        else:
            raise SystemExit(
                "PDF not found. Provide --pdf, --input-dir, or keep a single PDF in cwd."
            )

    if not pdf_paths:
        raise SystemExit(
            f"No PDF files found in directory: {input_dir}"
        )

    for pdf_path in pdf_paths:
        if not os.path.exists(pdf_path):
            raise SystemExit(f"PDF not found: {pdf_path}")

    if args.max_docs is not None and args.max_docs > 0:
        if len(pdf_paths) > args.max_docs:
            pdf_paths = pdf_paths[: args.max_docs]
            log(f"Limiting to first {args.max_docs} PDFs.")

    template_path = args.base_json or args.schema
    if not template_path:
        candidate_base = os.path.join(os.getcwd(), "JSON base.txt")
        if os.path.exists(candidate_base):
            template_path = candidate_base
        else:
            default_schema = os.path.join(os.path.dirname(__file__), "schema.json")
            if os.path.exists(default_schema):
                template_path = default_schema
            else:
                raise SystemExit(
                    "Template not found. Provide --base-json or --schema, or create schema.json."
                )

    if not os.path.exists(template_path):
        raise SystemExit(f"Template not found: {template_path}")
    log(f"Template: {template_path}")

    template = load_schema(template_path)

    vision_model = args.model or args.vision_model
    json_llm_state = None

    deepseek_state = None
    deepseek_cleanup_dir = None

    def get_deepseek_state():
        nonlocal deepseek_state, deepseek_cleanup_dir
        if deepseek_state is None:
            deepseek_state, deepseek_cleanup_dir = init_deepseek_ocr_state(
                vision_model,
                args.deepseek_base_size,
                args.deepseek_image_size,
                args.deepseek_crop_mode,
                max_new_tokens=args.deepseek_max_new_tokens,
                cache_dir=args.deepseek_cache_dir,
                attn_impl=args.deepseek_attn_impl,
                output_dir=args.deepseek_output_dir,
            )
            if deepseek_cleanup_dir:
                atexit.register(shutil.rmtree, deepseek_cleanup_dir, ignore_errors=True)
        return deepseek_state

    def get_json_llm_state():
        nonlocal json_llm_state
        if json_llm_state is None:
            if not args.json_llm_model:
                raise SystemExit("--json-llm requires --json-llm-model.")
            json_llm_state = load_json_llm(
                args.json_llm_model,
                cache_dir=args.json_llm_cache_dir,
                attn_impl=args.json_llm_attn_impl,
            )
        return json_llm_state

    if args.mode != "vision":
        log("Text mode is disabled; using vision-only DeepSeek pipeline.")

    multiple = len(pdf_paths) > 1
    excel_out = args.out_excel
    if multiple and not excel_out:
        excel_out = os.path.join(os.getcwd(), "extraction_results.xlsx")
    if excel_out:
        log(f"Excel output: {excel_out}")

    rows = []
    ocr_results = []
    excel_columns = list(flatten_json(template).keys())
    deepseek = get_deepseek_state()
    questionnaire_prompt = args.questionnaire_prompt or build_questionnaire_prompt()

    for index, pdf_path in enumerate(pdf_paths, start=1):
        log(f"Processing PDF ({index}/{len(pdf_paths)}): {pdf_path}")
        doc_start = time.perf_counter()
        total_pages = get_pdf_page_count(pdf_path)
        log(f"Found {total_pages} pages")

        page_numbers = parse_page_range(args.page_range, total_pages)
        log(f"Pages to process: {format_page_list(page_numbers)}")

        log(
            f"Rendering {len(page_numbers)} pages at {args.ocr_dpi} dpi (vision mode)..."
        )
        step_start = time.perf_counter()
        images = render_pdf_images(
            pdf_path,
            args.ocr_dpi,
            page_numbers=page_numbers,
            thread_count=args.render_threads,
            use_pdftocairo=args.use_pdftocairo,
        )
        log(
            f"Rendered images in {time.perf_counter() - step_start:.1f}s"
        )
        if not images:
            log(f"No pages rendered for {pdf_path}")
            if not multiple:
                raise SystemExit("No pages rendered from PDF.")
            continue

        text_pages = extract_text_pages_with_deepseek(
            images,
            page_numbers,
            deepseek,
            args.deepseek_prompt,
            args.image_format,
            args.image_quality,
            args.debug,
        )
        if not text_pages:
            log(f"No text extracted for {pdf_path}")
            if not multiple:
                raise SystemExit("No text extracted from PDF pages.")
            continue
        if args.ocr_output_dir:
            os.makedirs(args.ocr_output_dir, exist_ok=True)
            base_name = os.path.splitext(os.path.basename(pdf_path))[0]
            ocr_path = os.path.join(
                args.ocr_output_dir, f"{base_name}_ocr.txt"
            )
            with open(ocr_path, "w", encoding="utf-8") as f:
                for page_num, text in text_pages:
                    f.write(f"=== Page {page_num} ===\n")
                    f.write(text.rstrip())
                    f.write("\n\n")
            log(f"Wrote OCR output {ocr_path}")

        parsed = build_json_from_ocr(template, text_pages, pdf_path)
        if args.questionnaire_from_image:
            questionnaire_pages = find_questionnaire_pages(text_pages)
            checkbox_image_by_page = None
            if (
                args.checkbox_dpi
                and args.checkbox_dpi != args.ocr_dpi
                and questionnaire_pages
            ):
                log(
                    "Rendering questionnaire pages at "
                    f"{args.checkbox_dpi} dpi for checkbox detection..."
                )
                checkbox_images = render_pdf_images(
                    pdf_path,
                    args.checkbox_dpi,
                    page_numbers=questionnaire_pages,
                    thread_count=args.render_threads,
                    use_pdftocairo=args.use_pdftocairo,
                )
                if checkbox_images:
                    checkbox_image_by_page = dict(
                        zip(questionnaire_pages, checkbox_images)
                    )
                else:
                    log("No questionnaire pages rendered for checkbox detection.")
            image_by_page = dict(zip(page_numbers, images))
            update_questionnaire_from_image(
                parsed,
                text_pages,
                image_by_page,
                deepseek,
                questionnaire_prompt,
                args.image_format,
                args.image_quality,
                args.debug,
                checkbox_image_by_page=checkbox_image_by_page,
                questionnaire_pages=questionnaire_pages,
                checkbox_deskew=args.checkbox_deskew,
                checkbox_contrast=args.checkbox_contrast,
                checkbox_binarize=args.checkbox_binarize,
                checkbox_crop_left=args.checkbox_crop_left,
                checkbox_debug_dir=args.checkbox_debug_dir,
                fallback_to_ocr=args.questionnaire_ocr_fallback,
            )

        if args.json_llm:
            ocr_results.append(
                {
                    "pdf_path": pdf_path,
                    "text_pages": text_pages,
                    "parsed": parsed,
                }
            )
        else:
            out_path = resolve_output_json_path(args.out, pdf_path, multiple)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(parsed, f, ensure_ascii=False, indent=2)
            log(f"Wrote {out_path} in {time.perf_counter() - doc_start:.1f}s")
            print(f"Wrote {out_path}")
            rows.append(flatten_for_excel(parsed, excel_columns))

    if args.json_llm and ocr_results:
        log("OCR pass complete; releasing DeepSeek before JSON LLM.")
        release_deepseek_state(deepseek)
        deepseek = None
        if deepseek_cleanup_dir:
            shutil.rmtree(deepseek_cleanup_dir, ignore_errors=True)
            deepseek_cleanup_dir = None

        for index, item in enumerate(ocr_results, start=1):
            pdf_path = item["pdf_path"]
            text_pages = item["text_pages"]
            parsed = item["parsed"]
            log(f"JSON LLM pass ({index}/{len(ocr_results)}): {pdf_path}")
            llm_start = time.perf_counter()
            combined = ""
            for _, text in text_pages:
                combined += text + "\n"
            max_chars = args.json_llm_max_input_chars
            if max_chars is not None and max_chars <= 0:
                max_chars = None
            user_prompt = build_json_llm_user_prompt(
                template,
                combined,
                os.path.basename(pdf_path),
                max_chars=max_chars,
            )
            llm_state = get_json_llm_state()
            llm_raw = generate_json_with_llm(
                llm_state,
                JSON_LLM_SYSTEM_PROMPT,
                user_prompt,
                args.json_llm_max_new_tokens,
                args.json_llm_temperature,
            )
            if args.json_llm_debug_dir:
                os.makedirs(args.json_llm_debug_dir, exist_ok=True)
                debug_name = os.path.splitext(os.path.basename(pdf_path))[0]
                debug_path = os.path.join(
                    args.json_llm_debug_dir, f"{debug_name}_llm.json.txt"
                )
                with open(debug_path, "w", encoding="utf-8") as f:
                    f.write(llm_raw)
            llm_data = extract_json_from_text(llm_raw)
            merged = parsed
            if llm_data:
                merged = merge_missing(llm_data, parsed)

            out_path = resolve_output_json_path(args.out, pdf_path, multiple)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(merged, f, ensure_ascii=False, indent=2)
            log(f"Wrote {out_path} in {time.perf_counter() - llm_start:.1f}s")
            print(f"Wrote {out_path}")
            rows.append(flatten_for_excel(merged, excel_columns))

    if rows and excel_out:
        write_excel(rows, excel_out)
        print(f"Wrote {excel_out}")
        if args.open_excel:
            open_excel_file(excel_out)

    log(f"Done in {time.perf_counter() - start_time:.1f}s")


if __name__ == "__main__":
    main()
