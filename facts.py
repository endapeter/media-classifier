"""
Deterministic metadata extraction: EXIF datetime, GPS, camera, and
filename/path date fallbacks, plus offline reverse geocoding.

None of this involves the VLM.
"""

import re
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Optional, Set, Tuple

from PIL import Image

from config import FULL_DATE_RE, YEAR_RE
from helpers import valid_year
from records import ImageFacts, WhenInfo
from ui import print


# ============================================================
# Date/year extraction
# ============================================================

def parse_exif_datetime(value: Any) -> Optional[Tuple[datetime, bool]]:
    """
    Parse common EXIF date formats.

    Returns (datetime, time_known); time_known is False when the source
    string only carried a date (no time of day), so midnight should not be
    trusted as a real timestamp.
    """
    if not isinstance(value, str):
        return None

    value = value.strip()

    for fmt, has_time in (
        ("%Y:%m:%d %H:%M:%S", True),
        ("%Y-%m-%d %H:%M:%S", True),
        ("%Y:%m:%d", False),
        ("%Y-%m-%d", False),
    ):
        try:
            return datetime.strptime(value, fmt), has_time
        except ValueError:
            pass

    # Fallback regex for embedded YYYY-MM-DD / YYYY:MM:DD.
    m = re.search(r"((?:19|20)\d{2})[:\-/](\d{2})[:\-/](\d{2})", value)
    if m:
        try:
            return (
                datetime(int(m.group(1)), int(m.group(2)), int(m.group(3))),
                False,
            )
        except ValueError:
            return None

    return None


def get_exif_datetime(image: Image.Image) -> Optional[Tuple[datetime, bool]]:
    """
    Extract (datetime, time_known) from EXIF.

    Priority:
        DateTimeOriginal
        DateTimeDigitized
        DateTime
    """
    try:
        exif = image.getexif()

        if not exif:
            return None

        values = []

        # Base IFD tags.
        for tag in (36867, 36868, 306):  # DateTimeOriginal, DateTimeDigitized, DateTime
            try:
                if tag in exif:
                    values.append(exif[tag])
            except Exception:
                pass

        # Exif IFD tags.
        try:
            exif_ifd = exif.get_ifd(0x8769)
            for tag in (36867, 36868):
                if tag in exif_ifd:
                    values.append(exif_ifd[tag])
        except Exception:
            pass

        for value in values:
            parsed = parse_exif_datetime(value)
            if parsed and valid_year(parsed[0].year):
                return parsed

    except Exception:
        return None

    return None


def get_exif_gps(image: Image.Image) -> Optional[Tuple[float, float]]:
    """
    Extract GPS coordinates from the EXIF GPS IFD.

    Returns (latitude, longitude) in decimal degrees, or None when absent,
    invalid, or exactly (0, 0) (a common placeholder for "no fix").
    """

    def _dms_to_decimal(value: Any) -> Optional[float]:
        # GPS coordinates arrive as an IFDRational, a single float/int, or a
        # tuple of up to three rationals (degrees, minutes, seconds).
        try:
            if isinstance(value, (tuple, list)):
                parts = [float(p) for p in value[:3]]
            else:
                parts = [float(value)]

            parts += [0.0] * (3 - len(parts))

            return parts[0] + parts[1] / 60.0 + parts[2] / 3600.0
        except Exception:
            return None

    try:
        exif = image.getexif()

        if not exif:
            return None

        gps_ifd = exif.get_ifd(0x8825)

        if not gps_ifd:
            return None

        lat = _dms_to_decimal(gps_ifd.get(2))  # GPSLatitude
        lon = _dms_to_decimal(gps_ifd.get(4))  # GPSLongitude
        lat_ref = gps_ifd.get(1)  # N or S
        lon_ref = gps_ifd.get(3)  # E or W

        if lat is None or lon is None:
            return None

        if str(lat_ref).upper().startswith("S"):
            lat = -lat

        if str(lon_ref).upper().startswith("W"):
            lon = -lon

        if abs(lat) < 1e-6 and abs(lon) < 1e-6:
            return None

        return (round(lat, 5), round(lon, 5))

    except Exception:
        return None


def get_exif_camera(image: Image.Image) -> Optional[str]:
    """Extract 'Make Model' from EXIF, or None when absent."""
    try:
        exif = image.getexif()

        if not exif:
            return None

        parts = []

        for tag in (271, 272):  # Make, Model
            value = exif.get(tag)

            if isinstance(value, bytes):
                value = value.decode("utf-8", errors="ignore")

            if isinstance(value, str):
                value = value.strip()

                if value:
                    parts.append(value)

        if not parts:
            return None

        # Normalize whitespace and drop filesystem-unsafe characters.
        camera = re.sub(r"[<>:\"/\\|?*]+", " ", " ".join(parts))
        camera = re.sub(r"\s+", " ", camera).strip()

        return camera or None

    except Exception:
        return None


def extract_full_date_from_text(text: str) -> Optional[date]:
    """Extract a full date from text if present."""
    for m in FULL_DATE_RE.finditer(text):
        try:
            y = int(m.group(1))
            mo = int(m.group(2))
            d = int(m.group(3))

            dt = date(y, mo, d)

            if valid_year(y):
                return dt
        except ValueError:
            continue

    return None


def extract_year_from_text(text: str) -> Optional[int]:
    """Extract a plausible year from text."""
    for m in YEAR_RE.finditer(text):
        y = int(m.group(1))
        if valid_year(y):
            return y

    return None


def extract_image_facts(
    img_path: Path,
    image: Image.Image,
    source_path: Path,
) -> ImageFacts:
    """
    Harvest all deterministic metadata without using the VLM.

    Date/year priority:
        1. EXIF datetime (full timestamp when time of day is present)
        2. Full date in filename
        3. Full date in parent folders inside source directory
        4. Year in filename
        5. Year in parent folders inside source directory

    GPS and camera are read from EXIF alongside; city/country are filled in
    later by reverse geocoding.
    """
    gps = get_exif_gps(image)
    camera = get_exif_camera(image)

    # 1. EXIF.
    exif_dt = get_exif_datetime(image)
    if exif_dt:
        dt, time_known = exif_dt
        return ImageFacts(
            when=WhenInfo(
                year=dt.year,
                date_token=dt.strftime("%Y%m%d"),
                source="exif",
            ),
            dt=dt if time_known else None,
            gps=gps,
            camera=camera,
        )

    # 2. Full date in filename.
    full_date = extract_full_date_from_text(img_path.name)
    if full_date:
        return ImageFacts(
            when=WhenInfo(
                year=full_date.year,
                date_token=full_date.strftime("%Y%m%d"),
                source="filename",
            ),
            gps=gps,
            camera=camera,
        )

    # Restrict path-based search to folders inside the source directory.
    try:
        rel_parent_parts = img_path.parent.relative_to(source_path).parts
    except Exception:
        rel_parent_parts = img_path.parent.parts

    # 3. Full date in parent folder names.
    for part in reversed(rel_parent_parts):
        full_date = extract_full_date_from_text(part)
        if full_date:
            return ImageFacts(
                when=WhenInfo(
                    year=full_date.year,
                    date_token=full_date.strftime("%Y%m%d"),
                    source="path",
                ),
                gps=gps,
                camera=camera,
            )

    # 4. Year in filename.
    year = extract_year_from_text(img_path.name)
    if year:
        return ImageFacts(
            when=WhenInfo(
                year=year,
                date_token=f"{year:04d}",
                source="filename_year",
            ),
            gps=gps,
            camera=camera,
        )

    # 5. Year in parent folder names.
    for part in reversed(rel_parent_parts):
        year = extract_year_from_text(part)
        if year:
            return ImageFacts(
                when=WhenInfo(
                    year=year,
                    date_token=f"{year:04d}",
                    source="path_year",
                ),
                gps=gps,
                camera=camera,
            )

    return ImageFacts(
        when=WhenInfo(year=None, date_token="undated", source="none"),
        gps=gps,
        camera=camera,
    )


def get_filesystem_when(path: Path) -> WhenInfo:
    """Fallback using file modification time."""
    try:
        st = path.stat()
        dt = datetime.fromtimestamp(st.st_mtime)

        if valid_year(dt.year):
            return WhenInfo(
                year=dt.year,
                date_token=dt.strftime("%Y%m%d"),
                source="filesystem",
            )
    except Exception:
        pass

    return WhenInfo(year=None, date_token="undated", source="none")


# ============================================================
# Offline reverse geocoding (EXIF GPS -> city/country)
# ============================================================

# Optional soft dependency: reverse_geocoder ships a local GeoNames database
# (~85 MB) and works fully offline. When it is unavailable, geocoding simply
# no-ops and the raw coordinates are still recorded in the XMP sidecar.
try:
    import reverse_geocoder as _reverse_geocoder
except Exception as _rg_error:
    _reverse_geocoder = None
    _rg_error = _rg_error


def reverse_geocode_batch(
    coords: Set[Tuple[float, float]],
) -> Dict[Tuple[float, float], Dict[str, str]]:
    """
    Resolve GPS coordinates to city/country names in one offline batch.

    Returns a dict keyed by the original coordinate tuples; empty when the
    package is unavailable or the lookup fails for any reason.
    """
    if _reverse_geocoder is None or not coords:
        if _reverse_geocoder is None and coords:
            print(
                f"[Warning] reverse_geocoder unavailable ({_rg_error}); "
                f"{len(coords)} GPS location(s) will not be geocoded."
            )

        return {}

    try:
        ordered = list(coords)
        results = _reverse_geocoder.search(ordered, verbose=False)

        resolved: Dict[Tuple[float, float], Dict[str, str]] = {}

        for coord, result in zip(ordered, results):
            resolved[coord] = {
                "city": (result.get("name") or "").strip() or None,
                "country": (result.get("cc") or "").strip() or None,
            }

        return resolved

    except Exception as e:
        print(f"[Warning] Reverse geocoding failed: {e}")
        return {}
