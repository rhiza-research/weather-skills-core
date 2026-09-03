"""Provenance chain handling for weather-skill artifacts.

``weather_skills_history`` is a JSON-encoded append-only array of entries
(oldest first). Each entry has ``skill``, ``version``, ``args``, and ``input``.
"""

import hashlib
import html
import json
import re
import sys
from pathlib import Path

HISTORY_ATTR = "weather_skills_history"
SOURCE_ATTR = "weather_skills_source"
DEFAULT_SOFTWARE = "forecasting-skills"
OFFICIAL_MARK_TEXT = "weather-skills provenance verified"
# Circular rubber-stamp arcs (drawn uppercase for an inked look).
# Keep each arc short so letters stay large enough to read at corner size.
_MARK_ARC_TOP = "WEATHER-SKILLS"
_MARK_ARC_BOTTOM = "VERIFIED"
# Classic crimson rubber-stamp ink (RGBA) — opaque enough to read on maps.
_MARK_INK = (139, 15, 32, 250)

_EXIF_USER_COMMENT = 0x9286  # EXIF UserComment tag
_HTML_META_RE = re.compile(
    rf'<meta\s+name=["\']{re.escape(HISTORY_ATTR)}["\']\s+content=["\'](.*?)["\']\s*/?>',
    re.IGNORECASE | re.DOTALL,
)


def hash_zarr(zarr_path: Path) -> str:
    """Stable sha256 of a zarr directory's relative paths + file bytes."""
    zarr_path = Path(zarr_path)
    h = hashlib.sha256()
    for p in sorted(zarr_path.rglob("*")):
        if p.is_file():
            h.update(str(p.relative_to(zarr_path)).encode())
            h.update(p.read_bytes())
    return h.hexdigest()


def parse_chain(raw: str) -> list:
    """Strictly parse ``weather_skills_history`` JSON into a list."""
    try:
        chain = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        raise ValueError("value is not valid JSON") from None
    if not isinstance(chain, list):
        raise ValueError("value is not a JSON array")  # noqa: TRY004
    return chain


def coerce_chain(raw: str, label: str) -> list | None:
    """Lenient parse for render paths; warns and returns None if malformed."""
    try:
        chain = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        chain = None
    if not isinstance(chain, list):
        print(
            f"ignoring malformed weather_skills_history on {label}; "
            "run `provenance --check` for details",
            file=sys.stderr,
        )
        return None
    return chain


_ENTRY_KNOWN_KEYS = {"skill", "version", "args", "input"}
_INPUT_ITEM_KNOWN_KEYS = {"basename", "hash", "history"}


def _validate_input(value, loc: str, violations: list, notes: list) -> None:
    if value is None:
        return

    def _check_item(item, item_loc: str) -> None:
        if not isinstance(item, dict):
            violations.append(f"{item_loc}: input entry is not an object")
            return
        if "basename" not in item:
            violations.append(f"{item_loc}: missing required key 'basename'")
        elif not isinstance(item["basename"], str):
            violations.append(f"{item_loc}.basename: must be a string")
        if "hash" not in item:
            violations.append(f"{item_loc}: missing required key 'hash'")
        elif not isinstance(item["hash"], str):
            violations.append(f"{item_loc}.hash: must be a string")
        if "history" in item:
            _validate_chain(item["history"], f"{item_loc}.history", violations, notes)
        for key in item:
            if key not in _INPUT_ITEM_KNOWN_KEYS:
                notes.append(f"{item_loc}: unknown key {key!r}")

    if isinstance(value, list):
        for j, item in enumerate(value):
            _check_item(item, f"{loc}[{j}]")
        return
    if isinstance(value, dict):
        _check_item(value, loc)
        return
    violations.append(f"{loc}: must be null, an object, or an array of objects")


def _validate_chain(chain, loc: str, violations: list, notes: list) -> None:
    if not isinstance(chain, list):
        violations.append(f"{loc}: value is not a JSON array")
        return
    for i, entry in enumerate(chain):
        eloc = f"{loc}[{i}]"
        if not isinstance(entry, dict):
            violations.append(f"{eloc}: entry is not an object")
            continue
        if "skill" not in entry:
            violations.append(f"{eloc}: missing required key 'skill'")
        elif not isinstance(entry["skill"], str):
            violations.append(f"{eloc}.skill: must be a string")
        elif not entry["skill"]:
            violations.append(f"{eloc}.skill: must be a non-empty string")
        if "version" not in entry:
            violations.append(f"{eloc}: missing required key 'version'")
        elif not isinstance(entry["version"], str):
            violations.append(f"{eloc}.version: must be a string")
        if "args" not in entry:
            violations.append(f"{eloc}: missing required key 'args'")
        elif not isinstance(entry["args"], dict):
            violations.append(f"{eloc}.args: must be an object")
        if "input" not in entry:
            violations.append(f"{eloc}: missing required key 'input'")
        else:
            _validate_input(entry["input"], f"{eloc}.input", violations, notes)
        for key in entry:
            if key not in _ENTRY_KNOWN_KEYS:
                notes.append(f"{eloc}: unknown key {key!r}")


def validate_chain(chain, loc: str) -> tuple[list, list]:
    """Validate a parsed history chain. Returns ``(violations, notes)``."""
    violations: list = []
    notes: list = []
    _validate_chain(chain, loc, violations, notes)
    return violations, notes


def chain_is_intact(history) -> bool:
    """True if ``history`` is a non-empty schema-valid provenance chain."""
    if not isinstance(history, list) or not history:
        return False
    violations, _notes = validate_chain(history, HISTORY_ATTR)
    return not violations


def _load_mark_font(size: int):
    """Prefer a condensed display face so arc type stays readable at small diameter."""
    from PIL import ImageFont

    # Condensed first: the circle is small, and a wide bold face collides on the arc.
    candidates = (
        "/System/Library/Fonts/Supplemental/Arial Narrow Bold.ttf",
        "/System/Library/Fonts/Supplemental/DIN Condensed Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSansNarrow-Bold.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSansNarrow-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed-Bold.ttf",
        "/System/Library/Fonts/Supplemental/Impact.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "DejaVuSans-Bold.ttf",
        "DejaVuSans.ttf",
    )
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _paste_rotated_char(stamp, char, font, fill, cx, cy, angle_deg, *, scale: int):
    """Render one character, rotate it, and paste centered at ``(cx, cy)`` on ``stamp``."""
    from PIL import Image, ImageDraw

    # Oversized tile so rotated glyphs are not clipped.
    tile_size = max(48 * scale, int(getattr(font, "size", 12) * 3))
    tile = Image.new("RGBA", (tile_size, tile_size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(tile)
    bbox = draw.textbbox((0, 0), char, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    pos = (tile_size / 2 - bbox[0] - tw / 2, tile_size / 2 - bbox[1] - th / 2)
    # White halo for contrast on maps, then a same-ink stroke so stems stay thick.
    draw.text(
        pos,
        char,
        font=font,
        fill=fill,
        stroke_width=max(2, scale // 2),
        stroke_fill=(255, 255, 255, 230),
    )
    draw.text(
        pos,
        char,
        font=font,
        fill=fill,
        stroke_width=max(1, scale // 3),
        stroke_fill=fill,
    )
    rotated = tile.rotate(-angle_deg, resample=Image.Resampling.BICUBIC, expand=True)
    stamp.paste(rotated, (int(cx - rotated.width / 2), int(cy - rotated.height / 2)), rotated)


def _glyph_width(char: str, font) -> float:
    """Advance width of ``char`` in the given font."""
    getlength = getattr(font, "getlength", None)
    if getlength is not None:
        return max(1.0, float(getlength(char)))
    from PIL import Image, ImageDraw

    draw = ImageDraw.Draw(Image.new("L", (1, 1)))
    bbox = draw.textbbox((0, 0), char, font=font)
    return max(1.0, float(bbox[2] - bbox[0]))


def _draw_arc_text(stamp, text, cx, cy, radius, font, fill, *, top: bool, scale: int):
    """Draw ``text`` along a circular arc, spaced by glyph width (top over; bottom under)."""
    import math

    if not text:
        return
    tracking = max(float(scale) * 0.5, font.size * 0.08)
    widths = [_glyph_width(ch, font) for ch in text]
    total = sum(widths) + tracking * max(len(text) - 1, 0)
    # Cap the arc so type stays upright enough to read (~135°).
    span = min(total / max(radius, 1.0), 2.35)
    if top:
        start = -math.pi / 2 - span / 2
        direction = 1.0
    else:
        start = math.pi / 2 + span / 2
        direction = -1.0

    cursor = 0.0
    for char, width in zip(text, widths, strict=True):
        mid = cursor + width / 2.0
        angle = start + direction * (mid / total) * span
        cursor += width + tracking
        if char == " ":
            continue
        x = cx + radius * math.cos(angle)
        y = cy + radius * math.sin(angle)
        tangent_deg = math.degrees(angle) + (90.0 if top else -90.0)
        _paste_rotated_char(stamp, char, font, fill, x, y, tangent_deg, scale=scale)


def _render_circular_stamp(diameter: int):
    """Build a circular old-school rubber stamp as an RGBA image of size ``diameter``."""
    import math

    from PIL import Image, ImageDraw

    # Draw at 4× then downscale so condensed type stays sharp at corner size.
    scale = 4
    size = max(diameter, 32) * scale
    stamp = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(stamp)
    cx = cy = size / 2
    ink = _MARK_INK

    # Double ring — classic rubber-stamp silhouette (thin relative to diameter).
    outer_r = size / 2 - scale
    draw.ellipse(
        (cx - outer_r, cy - outer_r, cx + outer_r, cy + outer_r),
        outline=ink,
        width=max(2 * scale, size // 28),
    )
    inner_r = outer_r * 0.64
    draw.ellipse(
        (cx - inner_r, cy - inner_r, cx + inner_r, cy + inner_r),
        outline=ink,
        width=max(scale, size // 50),
    )

    font = _load_mark_font(max(10 * scale, int(size * 0.115)))
    text_r = (outer_r + inner_r) / 2
    _draw_arc_text(stamp, _MARK_ARC_TOP, cx, cy, text_r, font, ink, top=True, scale=scale)
    _draw_arc_text(stamp, _MARK_ARC_BOTTOM, cx, cy, text_r, font, ink, top=False, scale=scale)

    # Small center star — keep the rubber-stamp look without crowding the type.
    star_r = outer_r * 0.16
    pts = []
    for i in range(10):
        r = star_r if i % 2 == 0 else star_r * 0.42
        a = -math.pi / 2 + i * math.pi / 5
        pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    draw.polygon(pts, fill=ink)

    # Slight rotation so it looks hand-inked rather than UI chrome.
    stamp = stamp.rotate(-6, resample=Image.Resampling.BICUBIC, expand=True)
    out_w = max(1, round(stamp.width / scale))
    out_h = max(1, round(stamp.height / scale))
    return stamp.resize((out_w, out_h), resample=Image.Resampling.LANCZOS)


def _draw_official_mark(img):
    """Composite a circular ``weather-skills provenance verified`` rubber stamp onto ``img``.

    Placed bottom-right. No-ops (returns a copy) when the image is too small.
    """
    from PIL import Image

    w, h = img.size
    if min(w, h) < 96:
        return img.copy()

    # Compact corner mark; condensed type stays legible around ~12% of the short side.
    diameter = max(72, min(int(min(w, h) * 0.12), 96))
    stamp = _render_circular_stamp(diameter)
    margin = max(4, int(min(w, h) * 0.015))
    if stamp.width + 2 * margin > w or stamp.height + 2 * margin > h:
        return img.copy()

    base = img.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    x = w - margin - stamp.width
    y = h - margin - stamp.height
    overlay.paste(stamp, (x, y), stamp)

    marked = Image.alpha_composite(base, overlay)
    if img.mode == "RGBA":
        return marked
    # Stay in RGB after compositing. Converting a palette image back to ``P``
    # requantizes and can merge nearby fills (BoM IOD pink / blue both become
    # one purple).
    return marked.convert("RGB")


def load_history(zarr_path: Path) -> list:
    """Read a zarr store's history chain; empty on miss or malformation."""
    zarr_path = Path(zarr_path)
    try:
        import xarray as xr

        with xr.open_zarr(zarr_path, consolidated=False) as ds:
            raw = ds.attrs.get(HISTORY_ATTR)
    except (OSError, KeyError, ValueError):
        return []
    if not raw:
        return []
    parsed = coerce_chain(raw, str(zarr_path))
    return [] if parsed is None else parsed


def input_ref(path: Path) -> dict:
    """Single-input ``input`` value: ``{basename, hash}``."""
    path = Path(path)
    return {"basename": path.name, "hash": hash_zarr(path)}


def build_entry(skill: str, version: str, args: dict, input) -> dict:
    """Assemble a provenance entry."""
    return {"skill": skill, "version": version, "args": args, "input": input}


def stamp_zarr(ds, history: list, *, source: str | None = None) -> None:
    """Stamp history (and optional source) on a dataset; clear encodings."""
    ds.attrs[HISTORY_ATTR] = json.dumps(history, sort_keys=True)
    if source is not None:
        ds.attrs[SOURCE_ATTR] = source
    for v in ds.variables:
        ds[v].encoding = {}


def restamp_zarr(zarr_path: Path, history: list) -> None:
    """Rewrite history on an already-written zarr store in place."""
    import zarr

    group = zarr.open_group(str(zarr_path), mode="r+", use_consolidated=False)
    group.attrs[HISTORY_ATTR] = json.dumps(history, sort_keys=True)
    zarr.consolidate_metadata(str(zarr_path))


def stamp_figure(path: Path, history: list, *, software: str = DEFAULT_SOFTWARE) -> None:
    """Embed ``weather_skills_history`` into a PNG, JPEG, or HTML file.

    When the chain is intact (non-empty and schema-valid), also draw a circular
    old-school ``weather-skills provenance verified`` rubber stamp on PNG/JPEG
    pixels (bottom-right). HTML gets metadata only.
    """
    from weather_skills_core.errors import SkillError

    path = Path(path)
    payload = json.dumps(history, sort_keys=True)
    suffix = path.suffix.lower()
    mark = chain_is_intact(history)

    if suffix == ".png":
        from PIL import Image
        from PIL.PngImagePlugin import PngInfo

        with Image.open(path) as img:
            out = _draw_official_mark(img) if mark else img.copy()
            info = PngInfo()
            for key, value in img.info.items():
                if isinstance(value, str) and key not in (HISTORY_ATTR, "Software"):
                    info.add_text(key, value)
            info.add_text(HISTORY_ATTR, payload)
            info.add_text("Software", software)
            out.save(path, pnginfo=info)
        return

    if suffix in (".jpg", ".jpeg"):
        from PIL import Image

        with Image.open(path) as img:
            out = _draw_official_mark(img) if mark else img.copy()
            if out.mode not in ("RGB", "L"):
                out = out.convert("RGB")
            exif = img.getexif()
            # ASCII UserComment: 8-byte charset header + payload
            exif[_EXIF_USER_COMMENT] = b"ASCII\x00\x00\x00" + payload.encode("ascii")
            out.save(path, exif=exif, quality=95)
        return

    if suffix in (".html", ".htm"):
        text = path.read_text(encoding="utf-8")
        meta = f'<meta name="{HISTORY_ATTR}" content="{html.escape(payload, quote=True)}">'
        if _HTML_META_RE.search(text):
            text = _HTML_META_RE.sub(meta, text, count=1)
        elif re.search(r"<head[^>]*>", text, re.IGNORECASE):
            text = re.sub(r"(<head[^>]*>)", rf"\1\n{meta}", text, count=1, flags=re.IGNORECASE)
        else:
            text = meta + "\n" + text
        path.write_text(text, encoding="utf-8")
        return

    raise SkillError(
        f"unsupported figure type {suffix!r} for {path}; expected .png, .jpg/.jpeg, or .html/.htm"
    )


def load_figure_history(path: Path) -> list | None:
    """Read history from a stamped figure file, or None if absent."""
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".png":
        from PIL import Image

        with Image.open(path) as img:
            raw = img.info.get(HISTORY_ATTR)
        return coerce_chain(raw, path.name) if raw else None

    if suffix in (".jpg", ".jpeg"):
        from PIL import Image

        with Image.open(path) as img:
            raw = img.getexif().get(_EXIF_USER_COMMENT)
        if raw is None:
            return None
        if isinstance(raw, bytes):
            if raw.startswith(b"ASCII\x00\x00\x00"):
                raw = raw[8:].decode("ascii")
            else:
                raw = raw.decode("utf-8", errors="replace")
        return coerce_chain(raw, path.name)

    if suffix in (".html", ".htm"):
        text = path.read_text(encoding="utf-8")
        m = _HTML_META_RE.search(text)
        if not m:
            return None
        return coerce_chain(html.unescape(m.group(1)), path.name)

    return None
