"""
The main image organizer pipeline.

Phases:
    1. scan         recursively find images
    2. enrich       deterministic metadata (EXIF time/GPS/camera),
                    offline reverse geocoding
    3. embed        CLIP embeddings, cluster into events within year-month
    3b. name events one Qwen call per cluster viewing a contact sheet
    4. describe     one Qwen call per image -> structured slots
                    (with checkpoint resume)
    5. fill events  fallback event names from the images' own slots
    6. compose       folders + filenames in code, sequence numbers,
                    event preview
    7. write+move    move/copy files, write XMP sidecars
"""

import os
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

from PIL import Image

from config import (
    ALLOW_MODEL_YEAR_FALLBACK,
    ANALYSIS_CHECKPOINT,
    CHECKPOINT_SCHEMA,
    COPY_INSTEAD_OF_MOVE,
    DRY_RUN,
    IMAGE_EXTENSIONS,
    MAX_TOTAL_PATH_LENGTH,
    SKIP_HIDDEN,
    UNKNOWN_YEAR_FOLDER,
    USE_FILESYSTEM_DATE_FALLBACK,
)
from clustering import (
    cluster_images,
    embed_images,
    fallback_event_name,
    load_embedder,
    name_events,
)
from facts import (
    extract_image_facts,
    get_filesystem_when,
    reverse_geocode_batch,
)
from helpers import print_time_estimate, valid_year
from naming import (
    assign_sequence_numbers,
    choose_destination,
    compose_event_folder,
    compose_filename,
)
from records import ImageFacts, PlannedImage, WhenInfo
from state import (
    append_log,
    checkpoint_matches_file,
    load_checkpoint,
    load_index,
    save_checkpoint,
    save_index,
)
from ui import print
from vlm import (
    extract_observed_year,
    load_model,
    predict_metadata,
    prepare_for_inference,
)
from xmp import write_xmp_sidecar


def organize_image_library(source_dir: str, target_dir: str) -> None:
    source_path = Path(source_dir).expanduser().resolve()
    target_path = Path(target_dir).expanduser().resolve()

    if not source_path.exists():
        raise FileNotFoundError(f"Source directory does not exist: {source_path}")

    if source_path == target_path:
        raise ValueError("Source and output directory must be different.")

    # Only skip files inside target if target is inside source.
    # If target is a parent of source, we still want to process source files.
    try:
        target_is_inside_source = target_path.is_relative_to(source_path)
    except AttributeError:
        target_is_inside_source = str(target_path).startswith(str(source_path) + os.sep)

    if target_is_inside_source:
        print("Notice: output directory is inside source directory. Output files will be skipped during scanning.")

    # ------------------------------------------------------------
    # Find images recursively.
    # ------------------------------------------------------------

    image_files: List[Path] = []

    for p in sorted(source_path.rglob("*")):
        try:
            if not p.is_file():
                continue

            if p.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            # Avoid processing already organized output files.
            if target_is_inside_source:
                if p == target_path or target_path in p.parents:
                    continue

            if SKIP_HIDDEN:
                try:
                    rel_parts = p.relative_to(source_path).parts
                except ValueError:
                    rel_parts = p.parts

                if any(part.startswith(".") for part in rel_parts):
                    continue

            image_files.append(p)

        except OSError:
            continue

    print(f"Found {len(image_files)} images in {source_path}.\n")

    if not image_files:
        return

    processor, model = load_model()
    embed_model, embed_processor = load_embedder()

    index = {} if DRY_RUN else load_index(target_path)

    # Analysis checkpoint: reuse VLM results from an interrupted earlier
    # run. Entries whose source file no longer exists (already moved in an
    # earlier run, or removed by the user) are dropped.
    checkpoint: Dict[str, Dict[str, Any]] = {}

    if ANALYSIS_CHECKPOINT and not DRY_RUN:
        checkpoint = load_checkpoint(target_path)
        checkpoint = {k: v for k, v in checkpoint.items() if Path(k).exists()}

    if checkpoint:
        print(
            f"Found analysis checkpoint with {len(checkpoint)} "
            f"already-analyzed image(s). Resuming where the last run stopped.\n"
        )

    skipped_count = 0
    failed_count = 0

    # ------------------------------------------------------------
    # Phase 2: deterministic metadata for every image (EXIF time, GPS,
    # camera), without any VLM. The filesystem date is an early year hint
    # so clustering buckets work before the model has seen anything.
    # ------------------------------------------------------------

    pending: List[Tuple[Path, ImageFacts]] = []

    for idx, img_path in enumerate(image_files, start=1):
        try:
            source_key = str(img_path)

            # Resume support.
            if not DRY_RUN and source_key in index:
                existing_destination = Path(index[source_key])

                if existing_destination.exists():
                    skipped_count += 1
                    continue

            try:
                rel = img_path.relative_to(source_path)
            except ValueError:
                rel = img_path

            print(f"[{idx}/{len(image_files)}] Reading metadata: {rel}")

            try:
                with Image.open(img_path) as img:
                    facts = extract_image_facts(img_path, img, source_path)
            except Exception as e:
                failed_count += 1
                print(f"  [Error] Failed to process {img_path.name}: {e}")

                if not DRY_RUN:
                    append_log(
                        target_path,
                        {
                            "timestamp": datetime.now().isoformat(),
                            "source": str(img_path),
                            "error": str(e),
                            "phase": "metadata",
                        },
                    )

                continue

            # Filesystem dates are unreliable for copied files, so this is
            # only a bucketing hint; a model-observed year in Phase 4 still
            # overrides it (priority: deterministic > model > filesystem).
            if facts.when.year is None and USE_FILESYSTEM_DATE_FALLBACK:
                fs_when = get_filesystem_when(img_path)

                if fs_when.year is not None:
                    facts.when = fs_when

            pending.append((img_path, facts))

        except Exception as e:
            failed_count += 1

            try:
                shown = img_path.relative_to(source_path)
            except ValueError:
                shown = img_path

            print(f"  [Error] Unexpected failure while reading {shown}: {e}")

            if not DRY_RUN:
                append_log(
                    target_path,
                    {
                        "timestamp": datetime.now().isoformat(),
                        "source": str(img_path),
                        "error": str(e),
                        "phase": "metadata",
                    },
                )

    # Reverse-geocode the unique GPS points once, in one offline batch.
    if pending:
        geo_coords = {facts.gps for _, facts in pending if facts.gps is not None}

        if geo_coords:
            print(f"\nReverse geocoding {len(geo_coords)} unique GPS location(s)...")

            geocoded = reverse_geocode_batch(geo_coords)

            for _, facts in pending:
                match = geocoded.get(facts.gps) if facts.gps is not None else None

                if match:
                    facts.city = match.get("city")
                    facts.country = match.get("country")

    # ------------------------------------------------------------
    # Phase 3: embed every image with CLIP and cluster it into events
    # (never across year-month boundaries).
    # ------------------------------------------------------------

    print(f"\nEmbedding {len(pending)} image(s) for event clustering...")

    embeddings = embed_images(
        embed_model,
        embed_processor,
        [img_path for img_path, _ in pending],
    )

    cluster_ids = cluster_images([facts for _, facts in pending], embeddings)

    clusters: Dict[int, List[Path]] = {}

    for (img_path, _), cluster_id in zip(pending, cluster_ids):
        clusters.setdefault(cluster_id, []).append(img_path)

    print(f"\nFound {len(clusters)} event cluster(s).")

    # ------------------------------------------------------------
    # Phase 3b: one Qwen call per cluster, viewing a contact sheet of
    # its frames, names the event ("Newborn Photos").
    # ------------------------------------------------------------

    event_names = name_events(model, processor, clusters)

    # ------------------------------------------------------------
    # Phase 4: per-image structured VLM analysis (with checkpoint
    # resume). Filenames are composed from these slots later, in code.
    # ------------------------------------------------------------

    analyzed: List[PlannedImage] = []
    resumed_count = 0

    # Per-image analysis durations, used to estimate the remaining time.
    # Index 0 is the compile-heavy first image and is excluded from the
    # average; see print_time_estimate.
    image_durations: List[float] = []

    for idx, ((img_path, facts), cluster_id) in enumerate(
        zip(pending, cluster_ids),
        start=1,
    ):
        try:
            try:
                rel = img_path.relative_to(source_path)
            except ValueError:
                rel = img_path

            print(f"[{idx}/{len(pending)}] Analyzing: {rel}")

            source_key = str(img_path)

            # Checkpoint resume: reuse the analysis from an interrupted
            # earlier run and skip inference.
            cached_entry = checkpoint.get(source_key)

            if cached_entry is not None and checkpoint_matches_file(img_path, cached_entry):
                analyzed.append(
                    PlannedImage(
                        img_path=img_path,
                        facts=facts,
                        slots=cached_entry.get("slots") or {},
                        raw_response=cached_entry.get("raw_response") or "",
                        cluster_id=cluster_id,
                    )
                )

                print("  -> Analysis resumed from checkpoint (skipping inference)")
                resumed_count += 1
                continue

            analysis_start = time.perf_counter()

            try:
                with Image.open(img_path) as img:
                    # Letterbox onto the fixed inference canvas so every image
                    # produces identical input shapes; otherwise OpenVINO
                    # recompiles the graph for each new image geometry.
                    inference_image = prepare_for_inference(img.convert("RGB"))

                # ----------------------------------------------------
                # VLM inference (structured slots).
                # ----------------------------------------------------

                try:
                    slots, raw_response = predict_metadata(
                        model,
                        processor,
                        inference_image,
                    )
                except Exception as e:
                    print(
                        f"  [Warning] Model inference failed: {e}\n"
                        f"  Falling back to a generic naming for this image."
                    )
                    slots, raw_response = {}, ""

                # A model-observed year is the fallback of last resort, and
                # also overrides the early filesystem year hint.
                if ALLOW_MODEL_YEAR_FALLBACK:
                    observed_year = extract_observed_year(slots)

                    if observed_year is not None and valid_year(observed_year):
                        if (
                            facts.when.year is None
                            or facts.when.source == "filesystem"
                        ):
                            facts.when = WhenInfo(
                                year=observed_year,
                                date_token=f"{observed_year:04d}",
                                source="model",
                            )

                print(
                    f"  Date token: {facts.when.date_token} "
                    f"(source: {facts.when.source})"
                )

                image_durations.append(time.perf_counter() - analysis_start)
                print_time_estimate(image_durations, idx, len(pending))

                analyzed.append(
                    PlannedImage(
                        img_path=img_path,
                        facts=facts,
                        slots=slots,
                        raw_response=raw_response,
                        cluster_id=cluster_id,
                    )
                )

                # Checkpoint the analysis so an interrupted run can resume here.
                # Failed inference (empty raw_response) is not checkpointed, so
                # it is retried on the next run instead of keeping the fallback.
                if ANALYSIS_CHECKPOINT and raw_response:
                    try:
                        st = img_path.stat()

                        checkpoint[source_key] = {
                            "schema": CHECKPOINT_SCHEMA,
                            "size": st.st_size,
                            "mtime_ns": st.st_mtime_ns,
                            "slots": slots,
                            "raw_response": raw_response[:1000],
                        }

                        save_checkpoint(target_path, checkpoint)

                    except OSError:
                        pass

            except Exception as e:
                failed_count += 1
                print(f"  [Error] Failed to process {img_path.name}: {e}")

                if not DRY_RUN:
                    append_log(
                        target_path,
                        {
                            "timestamp": datetime.now().isoformat(),
                            "source": str(img_path),
                            "error": str(e),
                        },
                    )
        except Exception as e:
            failed_count += 1

            try:
                shown = img_path.relative_to(source_path)
            except ValueError:
                shown = img_path

            print(f"  [Error] Unexpected failure while analyzing {shown}: {e}")

            if not DRY_RUN:
                append_log(
                    target_path,
                    {
                        "timestamp": datetime.now().isoformat(),
                        "source": str(img_path),
                        "error": str(e),
                        "phase": "analysis",
                    },
                )

    # ------------------------------------------------------------
    # Phase 5: fill in missing event names from the images' own slots.
    # ------------------------------------------------------------

    for record in analyzed:
        if not record.event_name:
            record.event_name = event_names.get(record.cluster_id, "")

        if not record.event_name:
            record.event_name = fallback_event_name(record.slots)

    # ------------------------------------------------------------
    # Phase 6: compose folders and filenames, assign sequence numbers,
    # and preview the events before anything moves.
    # ------------------------------------------------------------

    moves: List[Dict[str, Any]] = []

    for record in analyzed:
        year_folder = (
            str(record.facts.when.year)
            if record.facts.when.year is not None
            else UNKNOWN_YEAR_FOLDER
        )
        event_folder = compose_event_folder(record)
        dest_folder = target_path / year_folder / event_folder
        filename = compose_filename(record)

        # Windows MAX_PATH guard.
        overflow = len(str(dest_folder / filename)) - MAX_TOTAL_PATH_LENGTH

        if overflow > 0:
            stem = Path(filename).stem
            ext = Path(filename).suffix
            keep = max(20, len(stem) - overflow)
            filename = stem[:keep].rstrip("_-") + ext

        moves.append(
            {
                "record": record,
                "year_folder": year_folder,
                "event_folder": event_folder,
                "dest_folder": dest_folder,
                "filename": filename,
            }
        )

    assign_sequence_numbers(moves)

    print("\nEvent preview:")

    event_sizes: Dict[Tuple[str, str], int] = {}

    for move in moves:
        key = (move["year_folder"], move["event_folder"])
        event_sizes[key] = event_sizes.get(key, 0) + 1

    for year_folder, event_folder in sorted(event_sizes):
        print(
            f"  {year_folder}/{event_folder} "
            f"— {event_sizes[(year_folder, event_folder)]} image(s)"
        )

    print()

    # ------------------------------------------------------------
    # Phase 7: move/copy files and write XMP sidecars.
    # ------------------------------------------------------------

    planned: Set[Path] = set()
    success_count = 0

    for idx, move in enumerate(moves, start=1):
        record = move["record"]
        img_path = record.img_path

        try:
            try:
                rel = img_path.relative_to(source_path)
            except ValueError:
                rel = img_path

            dest_folder = move["dest_folder"]

            dest_path = choose_destination(
                dest_folder=dest_folder,
                filename=move["filename"],
                planned=planned,
                img_path=img_path,
            )

            try:
                shown_destination = dest_path.relative_to(target_path)
            except ValueError:
                shown_destination = dest_path

            print(f"[{idx}/{len(moves)}] Organizing: {rel}")
            print(
                f"  Event folder: {move['year_folder']}/{move['event_folder']}"
            )
            print(f"  Planned destination: {shown_destination}")

            # ----------------------------------------------------
            # Move/copy file, then write its XMP sidecar.
            # ----------------------------------------------------

            if not DRY_RUN:
                dest_folder.mkdir(parents=True, exist_ok=True)

                if COPY_INSTEAD_OF_MOVE:
                    shutil.copy2(img_path, dest_path)
                    action = "copied"
                else:
                    shutil.move(str(img_path), str(dest_path))
                    action = "moved"

                index[str(img_path)] = str(dest_path)
                save_index(target_path, index)

                # The analysis is no longer needed once the file has moved.
                if ANALYSIS_CHECKPOINT:
                    checkpoint.pop(str(img_path), None)
                    save_checkpoint(target_path, checkpoint)

                # Written after the image lands, so a failure never strands
                # an orphan sidecar.
                write_xmp_sidecar(record, dest_path)

                append_log(
                    target_path,
                    {
                        "timestamp": datetime.now().isoformat(),
                        "source": str(img_path),
                        "destination": str(dest_path),
                        "action": action,
                        "when": record.facts.when.__dict__,
                        "event_folder": move["event_folder"],
                        "cluster_id": record.cluster_id,
                        "slots": record.slots,
                        "raw_response": record.raw_response[:1000],
                    },
                )

                print(f"  -> {action.capitalize()} to: {shown_destination}")

            planned.add(dest_path)
            success_count += 1
        except Exception as e:
            failed_count += 1
            print(f"  [Error] Failed to process {img_path.name}: {e}")

            if not DRY_RUN:
                append_log(
                    target_path,
                    {
                        "timestamp": datetime.now().isoformat(),
                        "source": str(img_path),
                        "error": str(e),
                    },
                )

    print("\nSummary:")
    print(f"  Processed/planned: {success_count}")
    print(f"  Skipped already processed: {skipped_count}")
    print(f"  Resumed from checkpoint: {resumed_count}")
    print(f"  Failed: {failed_count}")

    if DRY_RUN:
        print("\nDry run complete. No files were moved or copied.")
