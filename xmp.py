"""
XMP metadata sidecar writer (ISO 16684-1, Dublin Core, IPTC Extension).

One .xmp file is written beside each organized image (same basename,
.xmp extension — the sidecar convention read by Lightroom, digiKam, and
every XMP-aware DAM). Image bytes are never modified.
"""

import json
from pathlib import Path
from typing import Any, Optional
from xml.sax.saxutils import escape as _xml_escape

from config import CATEGORY_VOCABULARY
from helpers import words_from_text
from naming import sanitize_slot_words
from records import PlannedImage
from ui import warn


def _xmp_li(value: Any) -> str:
    """One rdf:li line for a bag/seq/alt value."""
    return f"<rdf:li>{_xml_escape(str(value))}</rdf:li>"


def compose_description_sentence(record: PlannedImage) -> str:
    """Human-readable dc:description sentence built from the slots."""
    subject = " ".join(sanitize_slot_words(record.slots.get("subject"), max_words=6))
    action = " ".join(sanitize_slot_words(record.slots.get("action"), max_words=4))
    setting = " ".join(sanitize_slot_words(record.slots.get("setting"), max_words=6))

    details = record.slots.get("details")

    if not isinstance(details, (list, tuple)):
        details = []

    sentence = subject or "Photo"

    if action:
        sentence += f" {action}"

    if setting:
        sentence += f" in {setting}"

    if details:
        clean = [str(d).strip() for d in details[:3] if str(d).strip()]

        if clean:
            sentence += f" ({'; '.join(clean)})"

    facts = record.facts

    if facts.city or facts.country:
        place = ", ".join(p for p in (facts.city, facts.country) if p)
        sentence += f" Taken in {place}."

    return sentence[0].upper() + sentence[1:] if sentence else "Photo"


def build_xmp_packet(record: PlannedImage, dest_path: Path) -> str:
    """
    Compose a valid XMP packet for an organized image.

    dc:title / dc:description / dc:subject carry the naming; IPTC
    LocationCreated carries the reverse-geocoded place; a custom mcs:
    namespace carries provenance (source path, original filename, full
    prediction JSON, GPS coordinates, date source).
    """
    facts = record.facts

    title = record.event_name or " ".join(
        sanitize_slot_words(record.slots.get("subject"), max_words=6)
    ) or dest_path.stem

    keywords = []

    category = str(record.slots.get("category") or "").strip().lower()
    if category in CATEGORY_VOCABULARY:
        keywords.append(category)

    keywords.extend(words_from_text(record.event_name))

    for value in (facts.city, facts.country, facts.camera):
        if value:
            keywords.extend(words_from_text(value))

    # Deduplicate while preserving order.
    keywords = list(dict.fromkeys(k for k in keywords if k))

    prediction_json = json.dumps(
        record.slots,
        ensure_ascii=False,
        sort_keys=True,
    )

    lines = [
        '<?xpacket begin="" id="W5M0MpCehiHzreSzNTczkc9d"?>',
        '<x:xmpmeta xmlns:x="adobe:ns:meta/" x:xmptk="media-classifier">',
        '<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">',
        '<rdf:Description rdf:about=""',
        '    xmlns:dc="http://purl.org/dc/elements/1.1/"',
        '    xmlns:Iptc4xmpExt="http://iptc.org/std/Iptc4xmpExt/2008-02-29/"',
        '    xmlns:mcs="https://media-classifier.local/schema/2/">',
        "  <dc:title>",
        "    <rdf:Alt>",
        f'      <rdf:li xml:lang="x-default">{_xml_escape(title)}</rdf:li>',
        "    </rdf:Alt>",
        "  </dc:title>",
        "  <dc:description>",
        "    <rdf:Alt>",
        f'      <rdf:li xml:lang="x-default">{_xml_escape(compose_description_sentence(record))}</rdf:li>',
        "    </rdf:Alt>",
        "  </dc:description>",
        "  <dc:subject>",
        "    <rdf:Bag>",
    ]

    lines.extend(f"      {_xmp_li(k)}" for k in keywords)

    lines.extend([
        "    </rdf:Bag>",
        "  </dc:subject>",
    ])

    if facts.city or facts.country:
        attrs = []

        if facts.city:
            attrs.append(f'Iptc4xmpExt:City="{_xml_escape(facts.city)}"')

        if facts.country:
            attrs.append(f'Iptc4xmpExt:CountryName="{_xml_escape(facts.country)}"')

        lines.extend([
            "  <Iptc4xmpExt:LocationCreated>",
            "    <rdf:Bag>",
            f"      <rdf:li {' '.join(attrs)}/>",
            "    </rdf:Bag>",
            "  </Iptc4xmpExt:LocationCreated>",
        ])

    source_path = _xml_escape(str(record.img_path))
    source_name = _xml_escape(record.img_path.name)
    gps = (
        f"{facts.gps[0]:.5f},{facts.gps[1]:.5f}"
        if facts.gps is not None
        else ""
    )

    lines.extend([
        f"  <mcs:SourcePath>{source_path}</mcs:SourcePath>",
        f"  <mcs:SourceFilename>{source_name}</mcs:SourceFilename>",
        f"  <mcs:DateSource>{_xml_escape(facts.when.source)}</mcs:DateSource>",
        f"  <mcs:GPS>{_xml_escape(gps)}</mcs:GPS>",
        f"  <mcs:Prediction>{_xml_escape(prediction_json)}</mcs:Prediction>",
        "  <mcs:ClusterId>",
        f"    <rdf:Seq>{_xmp_li(record.cluster_id)}</rdf:Seq>",
        "  </mcs:ClusterId>",
        "</rdf:Description>",
        "</rdf:RDF>",
        "</x:xmpmeta>",
        '<?xpacket end="w"?>',
        "",
    ])

    return "\n".join(lines)


def write_xmp_sidecar(record: PlannedImage, dest_path: Path) -> Optional[Path]:
    """
    Write the XMP sidecar beside the destination image.

    The sidecar uses the destination's basename with the extension replaced
    by .xmp. Failures are logged but never abort the move.
    """
    try:
        sidecar_path = dest_path.with_suffix(".xmp")

        packet = build_xmp_packet(record, dest_path)

        tmp_path = sidecar_path.with_suffix(".xmp.tmp")
        tmp_path.write_text(packet, encoding="utf-8")
        tmp_path.replace(sidecar_path)

        return sidecar_path

    except Exception as e:
        warn(f"could not write XMP sidecar: {e}")
        return None
