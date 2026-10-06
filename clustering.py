"""
Embeddings, clustering, and event naming.

Images are embedded with CLIP (CPU, milliseconds per image), clustered
within each year-month bucket, and each cluster gets ONE Qwen call viewing
a contact sheet of its frames to name the event ("Newborn Photos"). This
gives the naming model the context a single image lacks, and keeps theme
naming consistent across the whole event.
"""

import hashlib
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from sklearn.cluster import AgglomerativeClustering
from transformers import AutoProcessor

from config import (
    CATEGORY_VOCABULARY,
    CLUSTER_DISTANCE_THRESHOLD,
    CONTACT_SHEET_MAX_FRAMES,
    CONTACT_SHEET_THUMB_SIZE,
    EMBED_BATCH_SIZE,
    EMBED_INPUT_SIZE,
    EMBED_MODEL_ID,
)
from naming import sanitize_event_name, sanitize_slot_words
from records import ImageFacts
from ui import counter, fmt_int, warn
from vlm import (
    OVModelForVisualCausalLM,
    get_event_prompt,
    predict_metadata,
    prepare_for_inference,
)


def load_embedder() -> Tuple[Any, Any]:
    """Load the CLIP embedding model and its processor (CPU)."""
    from transformers import CLIPModel

    print(f"Loading embedding model {EMBED_MODEL_ID}...")

    processor = AutoProcessor.from_pretrained(EMBED_MODEL_ID)
    model = CLIPModel.from_pretrained(EMBED_MODEL_ID)
    model.eval()

    print("Embedding model loaded (CPU).")

    return model, processor


def embed_images(
    model: Any,
    processor: AutoProcessor,
    image_paths: List[Path],
) -> "np.ndarray":
    """
    Compute L2-normalized CLIP embeddings for every image.

    Failed opens become neutral black images so the embedding matrix always
    aligns with the input list by index.
    """
    if not image_paths:
        return np.zeros((0, 512), dtype=np.float32)

    size = EMBED_INPUT_SIZE
    all_features: List["np.ndarray"] = []

    for start in range(0, len(image_paths), EMBED_BATCH_SIZE):
        batch_paths = image_paths[start:start + EMBED_BATCH_SIZE]
        images = []

        for p in batch_paths:
            try:
                with Image.open(p) as img:
                    images.append(
                        img.convert("RGB").resize((size, size), Image.BILINEAR)
                    )
            except Exception:
                images.append(Image.new("RGB", (size, size)))

        inputs = processor(images=images, return_tensors="pt")

        with torch.no_grad():
            features = model.get_image_features(pixel_values=inputs["pixel_values"])

        features = features / features.norm(p=2, dim=-1, keepdim=True)
        all_features.append(features.cpu().numpy())

        done = min(start + EMBED_BATCH_SIZE, len(image_paths))
        print(f"Embedded {fmt_int(done)}/{fmt_int(len(image_paths))} images")

    return np.vstack(all_features)


def _cluster_bucket_key(facts: ImageFacts) -> Tuple[str, Optional[int], Optional[str]]:
    """
    Bucket key for clustering.

    Photos are only ever clustered inside the same year-month so images
    from different occasions never merge. Year-only (or undated) images form
    one bucket per year.
    """
    token = facts.when.date_token

    if re.fullmatch(r"\d{8}", token):
        return ("ym", facts.when.year, token[4:6])

    return ("y", facts.when.year, None)


def cluster_images(
    facts_list: List[ImageFacts],
    embeddings: "np.ndarray",
) -> List[int]:
    """
    Cluster images within each bucket; returns a cluster id per image.

    Singleton buckets get their own cluster. If clustering fails for a
    bucket, every image in it becomes its own cluster (naming still works).
    """
    labels: List[int] = [0] * len(facts_list)

    buckets: Dict[Tuple, List[int]] = {}

    for idx, facts in enumerate(facts_list):
        buckets.setdefault(_cluster_bucket_key(facts), []).append(idx)

    next_label = 0

    for key in sorted(buckets.keys(), key=str):
        idxs = buckets[key]

        if len(idxs) == 1:
            labels[idxs[0]] = next_label
            next_label += 1
            continue

        try:
            clustering = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=CLUSTER_DISTANCE_THRESHOLD,
                metric="cosine",
                linkage="average",
            )

            cluster_labels = clustering.fit_predict(embeddings[idxs])

        except Exception as e:
            warn(f"clustering failed for {key}: {e}")
            cluster_labels = list(range(len(idxs)))

        for i, label in zip(idxs, cluster_labels):
            labels[i] = next_label + int(label)

        next_label += len(set(int(l) for l in cluster_labels))

    return labels


def build_contact_sheet(image_paths: List[Path]) -> Image.Image:
    """
    Grid of evenly-sampled thumbnails on a fixed-size canvas, letterboxed
    onto the standard inference canvas so the Qwen graph never recompiles.
    """
    max_frames = min(CONTACT_SHEET_MAX_FRAMES, len(image_paths))

    if len(image_paths) <= max_frames:
        chosen = list(image_paths)
    else:
        step = len(image_paths) / max_frames
        chosen = [
            image_paths[min(int(i * step), len(image_paths) - 1)]
            for i in range(max_frames)
        ]

    thumb = CONTACT_SHEET_THUMB_SIZE
    cols = min(4, len(chosen))
    rows = max(1, math.ceil(len(chosen) / cols))

    canvas = Image.new("RGB", (cols * thumb, rows * thumb), (0, 0, 0))

    for cell, path in enumerate(chosen):
        col = cell % cols
        row = cell // cols

        try:
            with Image.open(path) as img:
                frame = img.convert("RGB")
        except Exception:
            frame = Image.new("RGB", (thumb, thumb))

        scale = min(thumb / frame.width, thumb / frame.height, 1.0)
        new_size = (max(1, round(frame.width * scale)), max(1, round(frame.height * scale)))
        frame = frame.resize(new_size, Image.BILINEAR)

        cell_canvas = Image.new("RGB", (thumb, thumb), (0, 0, 0))
        cell_canvas.paste(frame, ((thumb - new_size[0]) // 2, (thumb - new_size[1]) // 2))

        canvas.paste(cell_canvas, (col * thumb, row * thumb))

    return prepare_for_inference(canvas)


def fallback_event_name(slots: Dict[str, Any]) -> str:
    """Derive an event name from the image's own slots when the VLM call failed."""
    category = str(slots.get("category") or "").strip().lower()

    if category in CATEGORY_VOCABULARY and category != "other":
        return " ".join(w.capitalize() for w in (category, "photos"))

    setting = sanitize_slot_words(slots.get("setting"), max_words=2)

    if setting:
        return " ".join(w.capitalize() for w in setting + ["photos"])

    return ""


def cluster_signature(image_paths: List[Path]) -> str:
    """
    Stable signature of a cluster: the sorted (path, size, mtime) of its
    members, hashed.

    A rerun over unchanged files re-derives the same signature, so a cached
    event name can be reused; any changed, added, or removed member produces
    a different signature and the event is named again. Returns "" if any
    member cannot be stat'ed (cache cannot be used or written).
    """
    parts = []

    for p in sorted(image_paths, key=str):
        try:
            st = p.stat()
        except OSError:
            return ""

        parts.append(f"{p}|{st.st_size}|{st.st_mtime_ns}")

    if not parts:
        return ""

    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()


def name_events(
    model: OVModelForVisualCausalLM,
    processor: AutoProcessor,
    clusters: Dict[int, List[Path]],
    on_named=None,
) -> Dict[int, str]:
    """
    One Qwen call per cluster, viewing the cluster's contact sheet.

    Returns cluster id -> event name; clusters whose call fails are simply
    absent from the dict (their images fall back later). When on_named is
    given, it is called as on_named(cluster_id, name) as soon as each name
    is produced, so the caller can checkpoint it immediately.
    """
    names: Dict[int, str] = {}

    ordered = sorted(clusters.keys())

    for idx, cluster_id in enumerate(ordered, start=1):
        paths = clusters[cluster_id]

        try:
            sheet = build_contact_sheet(paths)
            prediction, _ = predict_metadata(
                model,
                processor,
                sheet,
                prompt=get_event_prompt(),
                max_new_tokens=32,
            )

            name = sanitize_event_name(
                prediction.get("event_name")
                or prediction.get("event")
                or prediction.get("name")
            )

            if name:
                names[cluster_id] = name

                if on_named is not None:
                    on_named(cluster_id, name)

                print(
                    f"{counter(idx, len(ordered))} {len(paths)} image(s) -> {name}"
                )
            else:
                warn("could not name this event; images will use a fallback")

        except Exception as e:
            warn(f"event naming failed: {e}")

    return names
