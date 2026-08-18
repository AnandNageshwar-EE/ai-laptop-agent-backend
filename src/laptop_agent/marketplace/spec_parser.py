"""Extracting structured specifications from a marketplace listing title.

A search API returns a title, a price and a link — not structured specs. Laptop
titles do, however, almost always state RAM and storage, because that is how the
category is merchandised:

    "Lenovo IdeaPad Slim 3 AMD Ryzen 5 7520U 15.6" (16GB/512GB SSD/Windows 11)"

This module turns that into a :class:`~laptop_agent.domain.product.LaptopSpecs`.

Three rules make this safe to do on untrusted text:

1. **Regex only.** No evaluation, no dynamic dispatch, bounded input length.
2. **Absence is not invention.** A value that cannot be extracted stays ``None``,
   which the constraint logic treats as unknown-and-therefore-failing for any
   mandatory requirement. The parser never guesses a plausible weight.
3. **Ranges are validated.** An extracted number outside a physically sensible
   range is discarded rather than clamped, because a title reading "1024GB RAM"
   is a parsing error, not a specification.
"""

from __future__ import annotations

import re
from typing import Final

from ..domain.product import LaptopSpecs

#: Titles longer than this are truncated before parsing.
MAX_TITLE_CHARS: Final = 400

# RAM: "16GB RAM", "(16GB/512GB", "16 GB DDR5"
_RAM: Final = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"(\d{1,3})\s*GB\s*(?:LP)?DDR\d?[XL]?\s*RAM",
        r"(\d{1,3})\s*GB\s*(?:LP)?DDR\d",
        r"(\d{1,3})\s*GB\s*(?:unified\s+)?(?:RAM|memory)",
        r"RAM[\s:]*(\d{1,3})\s*GB",
        r"\((\d{1,3})\s*GB\s*/",          # "(16GB/512GB SSD/..."
        r"(\d{1,3})\s*GB\s*/\s*\d+\s*(?:GB|TB)",
    )
)

# Storage: "512GB SSD", "1TB SSD", "/512GB/"
_STORAGE_GB: Final = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"(\d{3,4})\s*GB\s*(?:PCIe\s*)?(?:NVMe\s*)?(?:M\.?2\s*)?SSD",
        r"(\d{3,4})\s*GB\s*(?:SATA\s*)?HDD",
        r"/\s*(\d{3,4})\s*GB\b",
        r"(\d{3,4})\s*GB\s*(?:storage|eMMC)",
    )
)
_STORAGE_TB: Final = re.compile(r"(\d(?:\.\d)?)\s*TB\b", re.IGNORECASE)

#: Last-resort RAM detection: a bare "8GB" that is not a storage capacity.
#: The negative lookahead keeps "512GB SSD" out, and the size ceiling keeps
#: storage figures out even when the media word is missing.
_BARE_GB: Final = re.compile(
    r"(\d{1,4})\s*GB\b(?!\s*(?:SSD|HDD|NVMe|eMMC|storage|PCIe|M\.?2))",
    re.IGNORECASE,
)

_SSD: Final = re.compile(r"\bSSD\b|\bNVMe\b|\beMMC\b", re.IGNORECASE)
_HDD: Final = re.compile(r"\bHDD\b|\bhard\s*disk\b", re.IGNORECASE)

_SCREEN: Final = re.compile(
    r"(\d{2}(?:\.\d)?)\s*(?:-\s*)?(?:\"|''|inch(?:es)?\b|\bFHD\b|\bWUXGA\b|\bQHD\b)",
    re.IGNORECASE,
)
_WEIGHT: Final = re.compile(r"(\d(?:\.\d{1,2})?)\s*(?:kg|kilograms?)\b", re.IGNORECASE)
_BATTERY: Final = re.compile(
    r"(?:up\s*to\s*)?(\d{1,2}(?:\.\d)?)\s*(?:hours?|hrs?)\b", re.IGNORECASE
)
_REFRESH: Final = re.compile(r"(\d{2,3})\s*Hz\b", re.IGNORECASE)

_OS_PATTERNS: Final = (
    ("macos", re.compile(r"\bmac\s*OS\b|\bmacbook\b", re.IGNORECASE)),
    ("chromeos", re.compile(r"\bchrome\s*OS\b|\bchromebook\b", re.IGNORECASE)),
    ("windows", re.compile(r"\bwindows\s*\d*\b|\bwin\s*1[01]\b", re.IGNORECASE)),
    ("linux", re.compile(r"\b(?:linux|ubuntu|dos)\b", re.IGNORECASE)),
)

_CPU: Final = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\b(Intel\s+Core\s+(?:Ultra\s+)?i[3579][\w-]*)",
        r"\b(AMD\s+Ryzen\s+[3579]\s*[\w-]*)",
        r"\b(Apple\s+M[1-4](?:\s+(?:Pro|Max|Ultra))?)",
        r"\b(M[1-4](?:\s+(?:Pro|Max|Ultra))?)\s+chip",
        r"\b(Intel\s+(?:Celeron|Pentium|Core)\s*[\w-]*)",
        r"\b(Snapdragon\s+X\s*(?:Elite|Plus)?)",
        r"\b(M[1-4]\s+(?:Pro|Max|Ultra))\b",
    )
)

_GPU: Final = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\b(NVIDIA\s+(?:GeForce\s+)?RTX\s*\d{4}[\w\s]{0,6}?)(?:\s*GPU)?\b",
        r"\b(RTX\s*\d{4})\b",
        r"\b(GTX\s*\d{3,4})\b",
        r"\b(AMD\s+Radeon\s+RX\s*\d{3,4}[MX]?)\b",
        r"\b(Intel\s+(?:Iris\s+Xe|UHD|Arc)\s*\w*)\b",
        r"\b(Radeon\s+\d{3}M)\b",
    )
)
_DEDICATED_GPU: Final = re.compile(
    r"\bRTX\s*\d{4}\b|\bGTX\s*\d{3,4}\b|\bRadeon\s+RX\s*\d{3,4}\b|"
    r"\bArc\s+A\d{3}\b|\bMX\s*\d{3}\b",
    re.IGNORECASE,
)
_TOUCH: Final = re.compile(r"\btouch\s*screen\b|\btouch\b", re.IGNORECASE)


def _first_int(patterns: tuple[re.Pattern[str], ...], text: str) -> int | None:
    for pattern in patterns:
        match = pattern.search(text)
        if match:
            try:
                return int(float(match.group(1)))
            except (TypeError, ValueError):
                continue
    return None


def _first_str(patterns: tuple[re.Pattern[str], ...], text: str) -> str:
    for pattern in patterns:
        match = pattern.search(text)
        if match:
            return " ".join(match.group(1).split())[:64]
    return ""


def _bare_ram(text: str) -> int | None:
    """First bare GB figure small enough to be memory rather than storage."""
    for match in _BARE_GB.finditer(text):
        try:
            value = int(match.group(1))
        except ValueError:
            continue
        if 2 <= value <= 128:
            return value
    return None


def parse_specs(title: str, description: str = "") -> LaptopSpecs:
    """Best-effort structured specs. Unextractable fields stay ``None``.

    ``description`` is consulted only for fields a title rarely carries (weight,
    battery). Both are untrusted text; nothing here executes or interprets them
    beyond pattern matching.
    """
    text = f"{title} {description}"[:MAX_TITLE_CHARS]

    ram = _first_int(_RAM, text)
    if ram is None:
        ram = _bare_ram(text)
    if ram is not None and not (2 <= ram <= 256):
        ram = None

    storage = _first_int(_STORAGE_GB, text)
    if storage is None:
        terabytes = _STORAGE_TB.search(text)
        if terabytes:
            try:
                storage = int(float(terabytes.group(1)) * 1024)
            except ValueError:
                storage = None
    if storage is not None and not (32 <= storage <= 8192):
        storage = None

    # Only claim a storage type when the text names one.
    storage_type: str | None = None
    if storage is not None:
        if _SSD.search(text):
            storage_type = "ssd"
        elif _HDD.search(text):
            storage_type = "hdd"

    screen: float | None = None
    screen_match = _SCREEN.search(text)
    if screen_match:
        try:
            value = float(screen_match.group(1))
            screen = value if 8.0 <= value <= 20.0 else None
        except ValueError:
            screen = None

    weight: float | None = None
    weight_match = _WEIGHT.search(text)
    if weight_match:
        try:
            value = float(weight_match.group(1))
            weight = value if 0.2 < value <= 6.0 else None
        except ValueError:
            weight = None

    battery: float | None = None
    battery_match = _BATTERY.search(text)
    if battery_match:
        try:
            value = float(battery_match.group(1))
            battery = value if 1.0 <= value <= 30.0 else None
        except ValueError:
            battery = None

    refresh: int | None = None
    refresh_match = _REFRESH.search(text)
    if refresh_match:
        value = int(refresh_match.group(1))
        refresh = value if 30 <= value <= 480 else None

    operating_system: str | None = None
    for name, pattern in _OS_PATTERNS:
        if pattern.search(text):
            operating_system = name
            break

    return LaptopSpecs(
        ram_gb=ram,
        storage_gb=storage,
        storage_type=storage_type,  # type: ignore[arg-type]
        cpu=_first_str(_CPU, text),
        gpu=_first_str(_GPU, text),
        dedicated_gpu=bool(_DEDICATED_GPU.search(text)),
        screen_inches=screen,
        weight_kg=weight,
        battery_hours=battery,
        os=operating_system,  # type: ignore[arg-type]
        touchscreen=bool(_TOUCH.search(text)),
        refresh_rate_hz=refresh,
    )
