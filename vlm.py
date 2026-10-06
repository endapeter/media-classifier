"""
VLM prompting and parsing: the Qwen2.5-VL model, its prompts, and the
structured-slot inference helpers built on top of it.
"""

import json
import re
from typing import Any, Dict, Optional, Tuple

from PIL import Image

from config import (
    CATEGORY_VOCABULARY,
    DEVICE_PRIORITY,
    INFERENCE_CANVAS_HEIGHT,
    INFERENCE_CANVAS_WIDTH,
    MODEL_ID,
    YEAR_RE,
)
from helpers import valid_year
from ui import note, warn

from optimum.intel.openvino import OVModelForVisualCausalLM
from transformers import AutoProcessor


def prepare_for_inference(image: Image.Image) -> Image.Image:
    """
    Downscale and letterbox onto the fixed inference canvas.

    The image is scaled down to fit inside the canvas (aspect ratio
    preserved), snapped to a multiple of 28 pixels per side, then centered
    on a black canvas of exactly INFERENCE_CANVAS_WIDTH x
    INFERENCE_CANVAS_HEIGHT. Constant input dimensions mean a constant
    vision-token count, so the OpenVINO graph compiles once instead of
    once per image.
    """
    canvas_w = INFERENCE_CANVAS_WIDTH
    canvas_h = INFERENCE_CANVAS_HEIGHT

    scale = min(canvas_w / image.width, canvas_h / image.height, 1.0)

    new_w = max(28, (round(image.width * scale) // 28) * 28)
    new_h = max(28, (round(image.height * scale) // 28) * 28)

    resized = image.resize((new_w, new_h), Image.LANCZOS)

    canvas = Image.new("RGB", (canvas_w, canvas_h), (0, 0, 0))
    canvas.paste(resized, ((canvas_w - new_w) // 2, (canvas_h - new_h) // 2))

    return canvas


def get_vlm_prompt() -> str:
    categories = ", ".join(CATEGORY_VOCABULARY)
    return f"""Analyze this image and describe only what is actually visible in it.

Return ONLY one valid JSON object. Do not use Markdown. Do not add comments.

{{
  "category": "<exactly one of: {categories}>",
  "subject": "who or what the image mainly shows",
  "setting": "where it takes place",
  "action": "what is happening",
  "details": ["distinctive specifics"],
  "observed_year": null
}}

Rules:
1. "category" must be exactly one word from the list above.

2. "subject", "setting", "action" are each 2-6 lowercase words describing only
   what is actually visible. No dates, no numbers from timestamps.

   Good examples:
   subject: "newborn baby", setting: "crib with striped blanket", action: "sleeping"
   subject: "group of friends", setting: "rocky beach at sunset", action: "hiking"
   subject: "utility bill document", setting: "wooden desk", action: "lying flat"

3. "details" is a list of 1-3 short lowercase phrases with distinctive
   specifics: colors, notable objects, weather, readable text.

4. "observed_year" must be an integer only if a year is clearly visible in the image.
   Otherwise return null.
"""


def get_event_prompt() -> str:
    return """You are looking at a contact sheet of photos taken around the same time
at the same occasion. Name the event or occasion they belong to.

Return ONLY one valid JSON object. Do not use Markdown. Do not add comments.

{
  "event_name": "2-4 word name"
}

Rules:
1. Name the occasion, not the image contents: "Newborn Photos", "Beach
   Vacation", "Kitchen Renovation", "Birthday Party", "Road Trip".
2. No dates, no numbers, no camera terms. Capitalize words.
"""


def parse_json_response(response: str) -> Optional[Dict[str, Any]]:
    """
    Robustly parse JSON from model output.

    Handles:
    - Markdown fences
    - Extra surrounding text
    - Trailing commas
    """
    if not response:
        return None

    text = response.strip()

    # Remove common Markdown code fences.
    text = re.sub(r"```(?:json)?", "", text)
    text = text.replace("`", "")

    # Direct parse.
    try:
        return json.loads(text)
    except Exception:
        pass

    # Try first { to last }.
    start = text.find("{")
    end = text.rfind("}")

    if start != -1 and end != -1 and end > start:
        snippet = text[start:end + 1]

        try:
            return json.loads(snippet)
        except Exception:
            pass

        # Remove trailing commas before } or ].
        repaired = re.sub(r",\s*([}\]])", r"\1", snippet)

        try:
            return json.loads(repaired)
        except Exception:
            pass

        # Last resort: regex object extraction.
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            snippet = m.group(0)

            try:
                return json.loads(snippet)
            except Exception:
                pass

            repaired = re.sub(r",\s*([}\]])", r"\1", snippet)

            try:
                return json.loads(repaired)
            except Exception:
                return None

    return None


def predict_metadata(
    model: OVModelForVisualCausalLM,
    processor: AutoProcessor,
    image: Image.Image,
    prompt: Optional[str] = None,
    max_new_tokens: int = 128,
) -> Tuple[Dict[str, Any], str]:
    """
    Run VLM inference and return parsed JSON plus raw response.

    `prompt` defaults to the per-image structured-slots prompt; the event
    naming pass passes its own contact-sheet prompt.
    """
    if prompt is None:
        prompt = get_vlm_prompt()

    # Prefer Qwen-style chat template.
    try:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        # Fallback for older processor versions.
        text = prompt

    inputs = processor(images=image, text=text, return_tensors="pt")

    output_ids = model.generate(
        **inputs,
        # The expected JSON response is ~50-70 tokens; generation is
        # autoregressive, so every extra token adds a decode step.
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1,
    )

    # Important: decode only newly generated tokens.
    # Otherwise the prompt's example JSON may be mistakenly parsed.
    try:
        input_len = inputs["input_ids"].shape[-1]
        generated_ids = output_ids[:, input_len:]
    except Exception:
        generated_ids = output_ids

    response = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )[0].strip()

    data = parse_json_response(response) or {}

    return data, response


def extract_observed_year(data: Dict[str, Any]) -> Optional[int]:
    """Extract model-observed year if valid."""
    for key in ("observed_year", "year", "image_year"):
        if key not in data:
            continue

        raw = data[key]

        if raw is None:
            continue

        s = str(raw).strip()

        if s.lower() in {"", "none", "null"}:
            continue

        try:
            y = int(float(s))
        except Exception:
            m = YEAR_RE.search(s)
            if not m:
                continue

            y = int(m.group(1))

        if valid_year(y):
            return y

    return None


# ============================================================
# Model loading
# ============================================================

def load_model() -> Tuple[AutoProcessor, OVModelForVisualCausalLM]:
    print(f"Loading processor from {MODEL_ID}...")

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = None

    for device in DEVICE_PRIORITY:
        # Intel GPUs cap single allocations; large vision-encoder buffers
        # can exceed that cap unless large allocations are enabled. The
        # option only exists in newer OpenVINO builds, so try it first and
        # fall back to a plain load if the plugin rejects it.
        attempts = [{}]

        if device.startswith("GPU"):
            attempts.insert(0, {"ov::intel_gpu::hint::enable_large_allocations": True})

        for ov_config in attempts:
            print(f"Loading model on {device}...")

            try:
                model = OVModelForVisualCausalLM.from_pretrained(
                    MODEL_ID,
                    device=device,
                    ov_config=ov_config,
                )
                print(f"Model loaded on {device}.")
                break

            except Exception as e:
                warn(f"failed to load on {device}: {e}")

                if ov_config:
                    note("retrying without large-allocation hint...")
                else:
                    note("trying next device...")

        if model is not None:
            break

    if model is None:
        raise RuntimeError(
            f"Could not load the model on any device: {DEVICE_PRIORITY}"
        )

    return processor, model
