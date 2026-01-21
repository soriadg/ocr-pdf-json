import importlib.util
import json
import os
import tempfile
import warnings
import time

from .logging_utils import log


def configure_transformers_logging():
    try:
        from transformers import logging as hf_logging
    except Exception:
        return
    hf_logging.set_verbosity_error()


def configure_transformers_warnings():
    warning_patterns = [
        r"You are using a model of type .* to instantiate a model of type .*",
        r"Some weights of .* were not initialized from the model checkpoint.*",
        r"You should probably TRAIN this model.*",
        r"`do_sample` is set to `False`.*temperature` is set to `0.0`.*",
        r"The attention mask and the pad token id were not set.*",
        r"Setting `pad_token_id` to `eos_token_id`.*",
        r"The `seen_tokens` attribute is deprecated.*",
        r"`get_max_cache\(\)` is deprecated.*",
        r"The attention layers in this model are transitioning.*",
    ]
    for pattern in warning_patterns:
        warnings.filterwarnings("ignore", message=pattern)


def ensure_deepseek_ocr_deps():
    required = {
        "torch": "torch",
        "transformers": "transformers",
        "torchvision": "torchvision",
        "einops": "einops",
        "addict": "addict",
        "easydict": "easydict",
        "matplotlib": "matplotlib",
        "tqdm": "tqdm",
        "numpy": "numpy",
        "requests": "requests",
    }
    missing = [
        pip_name
        for module, pip_name in required.items()
        if importlib.util.find_spec(module) is None
    ]
    if missing:
        packages = " ".join(sorted(set(missing)))
        raise SystemExit(
            "Missing DeepSeek-OCR dependencies. Install with:\n"
            f"  python -m pip install {packages}"
        )

    try:
        import transformers
        from transformers.models.llama import modeling_llama
    except Exception as exc:
        raise SystemExit(
            "DeepSeek-OCR requires transformers==4.46.3. "
            "Install with: python -m pip install 'transformers==4.46.3'"
        ) from exc

    if not hasattr(modeling_llama, "LlamaFlashAttention2"):
        raise SystemExit(
            f"DeepSeek-OCR requires transformers==4.46.3 (current: {transformers.__version__}). "
            "Install with: python -m pip install 'transformers==4.46.3'"
        )

    configure_transformers_logging()
    configure_transformers_warnings()


def normalize_deepseek_prompt(prompt):
    prompt = (prompt or "").strip()
    if not prompt:
        prompt = "Free OCR."
    if "<image>" not in prompt:
        prompt = "<image>\n" + prompt
    return prompt


def find_local_deepseek_model():
    candidates = [
        os.path.join(os.getcwd(), "models", "DeepSeek-OCR"),
        os.path.join(os.path.dirname(__file__), "models", "DeepSeek-OCR"),
    ]
    for path in candidates:
        if os.path.isdir(path):
            return path
    return None


def find_missing_model_files(model_dir):
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(os.path.join(model_dir, "model.safetensors")):
        return []
    if os.path.exists(os.path.join(model_dir, "pytorch_model.bin")):
        return []
    if not os.path.exists(index_path):
        return ["model.safetensors.index.json"]
    try:
        with open(index_path, "r", encoding="utf-8") as f:
            index = json.load(f)
    except (OSError, json.JSONDecodeError):
        return []
    weight_map = index.get("weight_map") or {}
    missing = []
    for filename in sorted(set(weight_map.values())):
        if not os.path.exists(os.path.join(model_dir, filename)):
            missing.append(filename)
    return missing


def load_deepseek_ocr_model(model_id, cache_dir=None, attn_impl=None):
    ensure_deepseek_ocr_deps()
    import torch
    from transformers import AutoModel, AutoTokenizer

    if os.path.isdir(model_id):
        missing = find_missing_model_files(model_id)
        if missing:
            missing_list = ", ".join(missing[:5])
            more = "" if len(missing) <= 5 else f" (+{len(missing) - 5} more)"
            raise SystemExit(
                "DeepSeek-OCR model files are missing. Re-download the model into "
                f"{model_id}.\nMissing: {missing_list}{more}\n"
                "Download command:\n"
                "  python -c \"from huggingface_hub import snapshot_download; "
                f"snapshot_download(repo_id='deepseek-ai/DeepSeek-OCR', "
                f"local_dir=r'{model_id}', resume_download=True)\""
            )

    if not torch.cuda.is_available():
        raise SystemExit(
            "DeepSeek-OCR requires a CUDA GPU. Install a CUDA-enabled torch build "
            "and run on a GPU host."
        )

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        cache_dir=cache_dir,
    )

    model_kwargs = {
        "trust_remote_code": True,
        "use_safetensors": True,
    }
    if attn_impl:
        model_kwargs["_attn_implementation"] = attn_impl
    if cache_dir:
        model_kwargs["cache_dir"] = cache_dir

    model = AutoModel.from_pretrained(model_id, **model_kwargs)
    model = model.eval().cuda().to(torch.bfloat16)
    return model, tokenizer


def wrap_deepseek_generate(model, max_new_tokens):
    if not max_new_tokens or max_new_tokens <= 0:
        return model
    original_generate = model.generate

    def generate_wrapper(*args, **kwargs):
        if "max_new_tokens" in kwargs:
            kwargs["max_new_tokens"] = min(kwargs["max_new_tokens"], max_new_tokens)
        else:
            kwargs["max_new_tokens"] = max_new_tokens
        return original_generate(*args, **kwargs)

    model.generate = generate_wrapper
    return model


def init_deepseek_ocr_state(
    model_id,
    base_size,
    image_size,
    crop_mode,
    max_new_tokens=None,
    cache_dir=None,
    attn_impl=None,
    output_dir=None,
):
    cleanup_dir = None
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        work_dir = output_dir
    else:
        work_dir = tempfile.mkdtemp(prefix="deepseek_ocr_")
        cleanup_dir = work_dir

    image_dir = os.path.join(work_dir, "input_images")
    os.makedirs(image_dir, exist_ok=True)

    model, tokenizer = load_deepseek_ocr_model(
        model_id,
        cache_dir=cache_dir,
        attn_impl=attn_impl,
    )
    if max_new_tokens and max_new_tokens > 0:
        wrap_deepseek_generate(model, max_new_tokens)

    state = {
        "model": model,
        "tokenizer": tokenizer,
        "base_size": base_size,
        "image_size": image_size,
        "crop_mode": crop_mode,
        "output_dir": work_dir,
        "image_dir": image_dir,
    }
    return state, cleanup_dir


def deepseek_ocr_image(state, prompt, image, image_format, image_quality, page_num):
    prompt = normalize_deepseek_prompt(prompt)
    fmt = image_format.lower()
    if fmt == "png":
        ext = "png"
        image_path = os.path.join(state["image_dir"], f"page_{page_num}.{ext}")
        image.save(image_path, format="PNG")
    else:
        ext = "jpg"
        image_path = os.path.join(state["image_dir"], f"page_{page_num}.{ext}")
        rgb = image.convert("RGB")
        rgb.save(image_path, format="JPEG", quality=image_quality, optimize=True)

    try:
        output = state["model"].infer(
            state["tokenizer"],
            prompt=prompt,
            image_file=image_path,
            output_path=state["output_dir"],
            base_size=state["base_size"],
            image_size=state["image_size"],
            crop_mode=state["crop_mode"],
            save_results=False,
            test_compress=False,
            eval_mode=True,
        )
    finally:
        try:
            os.remove(image_path)
        except OSError:
            pass

    return output or ""


def extract_text_pages_with_deepseek(
    images,
    page_numbers,
    deepseek_state,
    prompt,
    image_format,
    image_quality,
    debug,
):
    results = []
    for page_num, image in zip(page_numbers, images):
        log(f"DeepSeek OCR page {page_num} start")
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
            f"DeepSeek OCR page {page_num} done in "
            f"{time.perf_counter() - call_start:.1f}s "
            f"({len(response)} chars)"
        )
        if debug:
            debug_path = f"deepseek_ocr_page_{page_num}.txt"
            with open(debug_path, "w", encoding="utf-8") as f:
                f.write(response)
        cleaned = response.strip()
        if cleaned:
            results.append((page_num, cleaned))
    return results


def release_deepseek_state(state):
    if not state:
        return
    model = state.get("model")
    tokenizer = state.get("tokenizer")
    state["model"] = None
    state["tokenizer"] = None
    try:
        import torch
    except Exception:
        return
    try:
        if model is not None:
            del model
        if tokenizer is not None:
            del tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
