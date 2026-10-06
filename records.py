"""
Shared data structures passed between the pipeline phases.
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


@dataclass
class WhenInfo:
    year: Optional[int]
    date_token: str
    source: str


@dataclass
class ImageFacts:
    """
    Deterministic metadata harvested from the file and its EXIF, without any
    VLM involvement.

    `dt` carries the full timestamp when the time of day is actually known
    (EXIF only); otherwise only `when` (year/date token) is populated and the
    filename omits the time segment.
    """

    when: WhenInfo
    dt: Optional[datetime] = None
    gps: Optional[Tuple[float, float]] = None
    camera: Optional[str] = None
    city: Optional[str] = None
    country: Optional[str] = None


@dataclass
class PlannedImage:
    """An analyzed image awaiting its final folder assignment and move."""

    img_path: Path
    facts: ImageFacts
    slots: Dict[str, Any]
    raw_response: str
    cluster_id: int = -1
    event_name: str = ""

    @property
    def when(self) -> WhenInfo:
        return self.facts.when
