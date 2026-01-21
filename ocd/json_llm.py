import json
import re

from .logging_utils import log


SYSTEM_PROMPT = (
    "You extract structured data from OCR text. "
    "Return only valid JSON that matches the provided template exactly. "
    "Do not add or remove keys."
)


def build_user_prompt(template, ocr_text, source_name, max_chars=None):
    if max_chars and len(ocr_text) > max_chars:
        ocr_text = ocr_text[:max_chars]
        log(f"OCR text truncated to {max_chars} chars for JSON LLM.")
    template_json = json.dumps(template, ensure_ascii=True, indent=2)
    instructions = [
        "Fill the JSON template using the OCR text.",
        "Use null for unknown numeric or boolean values.",
        "Use empty string for unknown text values.",
        "Use empty arrays for unknown list values.",
        "For cuestionario.2_experiencia.d_empresa_exporta.exporta use 'si' or 'no'.",
        "All other yes/no fields should be true or false.",
        "Return ONLY JSON. No markdown.",
    ]
    return (
        "SOURCE: "
        + source_name
        + "\nINSTRUCTIONS:\n- "
        + "\n- ".join(instructions)
        + "\nTEMPLATE:\n"
        + template_json
        + "\nOCR:\n"
        + ocr_text
    )


def extract_json_from_text(text):
    if not text:
        return None
    cleaned = text.strip()
    if "```" in cleaned:
        cleaned = cleaned.replace("```json", "").replace("```", "").strip()
    first = cleaned.find("{")
    last = cleaned.rfind("}")
    if first == -1 or last == -1 or last <= first:
        return None
    cleaned = cleaned[first : last + 1]
    cleaned = re.sub(r",\s*([}\]])", r"\1", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return None


def load_json_llm(model_id, cache_dir=None, attn_impl=None, device=None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = torch.bfloat16 if device == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        cache_dir=cache_dir,
    )
    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch_dtype,
    }
    if attn_impl:
        model_kwargs["_attn_implementation"] = attn_impl
    if cache_dir:
        model_kwargs["cache_dir"] = cache_dir
    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    model = model.eval().to(device)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return {"model": model, "tokenizer": tokenizer, "device": device}


def generate_json_with_llm(
    state,
    system_prompt,
    user_prompt,
    max_new_tokens,
    temperature,
):
    tokenizer = state["tokenizer"]
    model = state["model"]
    device = state["device"]

    if hasattr(tokenizer, "apply_chat_template"):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        )
    else:
        prompt = system_prompt + "\n\n" + user_prompt
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    input_ids = input_ids.to(device)
    do_sample = temperature is not None and temperature > 0
    generate_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        generate_kwargs["temperature"] = temperature

    output_ids = model.generate(input_ids, **generate_kwargs)
    generated = output_ids[0][input_ids.shape[1] :]
    return tokenizer.decode(generated, skip_special_tokens=True)
