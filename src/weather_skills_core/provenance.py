"""Provenance graph handling for weather-skill artifacts.

``weather_skills_history`` is a JSON-encoded array of entries. A single-input
path stays oldest-first (a path DAG). A multi-input skill records a join: the
top-level array is just that skill's entry, and every parent subgraph lives
under ``input[].history``. Each new entry also records the git ``commit``
(and optional ``repo`` / ``dirty``) of the skill that ran.
"""

import hashlib
import html
import inspect
import json
import re
import subprocess
import sys
from pathlib import Path

HISTORY_ATTR = "weather_skills_history"
SOURCE_ATTR = "weather_skills_source"
DEFAULT_SOFTWARE = "forecasting-skills"
OFFICIAL_MARK_TEXT = "weather-skills provenance verified"
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


_ENTRY_KNOWN_KEYS = {"skill", "version", "args", "input", "commit", "repo", "dirty"}
_INPUT_ITEM_KNOWN_KEYS = {"basename", "hash", "history"}
_GIT_TIMEOUT_SECONDS = 2
_SSH_GITHUB_RE = re.compile(r"^git@github\.com:([^/]+)/(.+?)(?:\.git)?$")
_HTTPS_GITHUB_RE = re.compile(r"^https://github\.com/([^/]+)/(.+?)(?:\.git)?(?:/)?$")


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
        if "commit" in entry:
            if not isinstance(entry["commit"], str) or not entry["commit"]:
                violations.append(f"{eloc}.commit: must be a non-empty string")
        if "repo" in entry:
            if not isinstance(entry["repo"], str) or not entry["repo"]:
                violations.append(f"{eloc}.repo: must be a non-empty string")
        if "dirty" in entry and not isinstance(entry["dirty"], bool):
            violations.append(f"{eloc}.dirty: must be a boolean")
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


def _render_circular_stamp(diameter: int):
    """Build a small circular rubber stamp: ring + center star, no text."""
    import math

    from PIL import Image, ImageDraw

    scale = 4
    size = max(diameter, 24) * scale
    stamp = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(stamp)
    cx = cy = size / 2
    ink = _MARK_INK

    outer_r = size / 2 - scale
    draw.ellipse(
        (cx - outer_r, cy - outer_r, cx + outer_r, cy + outer_r),
        outline=ink,
        width=max(2 * scale, size // 22),
    )

    star_r = outer_r * 0.48
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

    # Compact star-only seal; ~6% of the short side.
    diameter = max(36, min(int(min(w, h) * 0.06), 48))
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


def input_ref(path: Path, history=None) -> dict:
    """One parent ``input`` value: ``{basename, hash}`` plus optional ``history``."""
    path = Path(path)
    ref = {"basename": path.name, "hash": hash_zarr(path)}
    if history is not None:
        ref["history"] = list(history)
    return ref


def input_items(entry) -> list:
    """Normalize an entry's ``input`` to a list of parent objects."""
    if not isinstance(entry, dict):
        return []
    value = entry.get("input")
    if value is None:
        return []
    if isinstance(value, dict):
        return [value]
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    return []


def parent_histories(entry) -> list[tuple[str, list]]:
    """``(basename, nested history)`` for every parent that recorded a subgraph."""
    out = []
    for item in input_items(entry):
        history = item.get("history")
        if isinstance(history, list):
            out.append((item.get("basename") or "?", history))
    return out


def origin_skill(history) -> str | None:
    """Oldest skill name, walking into the first nested parent of a join."""
    if not isinstance(history, list) or not history or not isinstance(history[0], dict):
        return None
    first = history[0]
    if len(history) == 1:
        for _name, nested in parent_histories(first):
            found = origin_skill(nested)
            if found:
                return found
    skill = first.get("skill")
    if isinstance(skill, str) and skill.strip():
        return skill.strip()
    return None


def normalize_repo_url(url: str) -> str:
    """Turn a git remote into an https clone URL when we recognize GitHub."""
    text = url.strip()
    ssh = _SSH_GITHUB_RE.match(text)
    if ssh:
        return f"https://github.com/{ssh[1]}/{ssh[2].removesuffix('.git')}"
    https = _HTTPS_GITHUB_RE.match(text)
    if https:
        return f"https://github.com/{https[1]}/{https[2].removesuffix('.git')}"
    return text.removesuffix(".git")


def _git(args: list[str], cwd: Path) -> str | None:
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False,
            timeout=_GIT_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip() or None


def _direct_url_payload(dist) -> dict | None:
    info_dir = getattr(dist, "_path", None)
    if info_dir is not None:
        candidate = Path(info_dir) / "direct_url.json"
        if candidate.is_file():
            try:
                payload = json.loads(candidate.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                payload = None
            else:
                if isinstance(payload, dict):
                    return payload
    for file in getattr(dist, "files", None) or []:
        if Path(str(file)).name != "direct_url.json":
            continue
        try:
            payload = json.loads(Path(file.locate()).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, ValueError):
            return None
        return payload if isinstance(payload, dict) else None
    return None


def _revision_from_direct_url(source_file: Path) -> dict | None:
    """Read PEP 610 ``direct_url.json`` for the installed dist that owns ``source_file``."""
    try:
        from importlib.metadata import distributions
    except ImportError:
        return None
    source_file = source_file.resolve()
    for dist in distributions():
        owned = False
        for file in getattr(dist, "files", None) or []:
            try:
                if Path(file.locate()).resolve() == source_file:
                    owned = True
                    break
            except (OSError, ValueError):
                continue
        if not owned:
            continue
        payload = _direct_url_payload(dist)
        if payload is None:
            continue
        vcs = payload.get("vcs_info") if isinstance(payload.get("vcs_info"), dict) else {}
        commit = vcs.get("commit_id")
        if not isinstance(commit, str) or not commit:
            continue
        revision = {"commit": commit}
        url = payload.get("url")
        if isinstance(url, str) and url:
            revision["repo"] = normalize_repo_url(url)
        return revision
    return None


def resolve_skill_revision(source=None) -> dict | None:
    """Git identity of the skill that is running.

    ``source`` may be a skill function or a filesystem path. Prefers the git
    checkout that contains the file; falls back to the installed package's
    ``direct_url.json`` (``uvx --from git+…@sha``). Returns ``None`` when
    neither is available. Untracked files do not count as dirty.
    """
    path = None
    if source is not None and not isinstance(source, (str, Path)):
        try:
            path = Path(inspect.getfile(source))
        except (OSError, TypeError):
            path = None
    elif source is not None:
        path = Path(source)
    if path is None:
        return None
    try:
        path = path.resolve()
    except OSError:
        return None
    cwd = path.parent if path.is_file() else path
    root = _git(["rev-parse", "--show-toplevel"], cwd)
    if not root:
        return _revision_from_direct_url(path) if path.is_file() else None
    root = Path(root)
    commit = _git(["rev-parse", "HEAD"], root)
    if not commit:
        return None
    revision = {"commit": commit}
    remote = _git(["config", "--get", "remote.origin.url"], root)
    if remote:
        revision["repo"] = normalize_repo_url(remote)
    porcelain = _git(["status", "--porcelain", "-uno"], root)
    if porcelain:
        revision["dirty"] = True
    return revision


def build_entry(skill: str, version: str, args: dict, input, revision=None) -> dict:
    """Assemble a provenance entry, attaching git identity when known."""
    entry = {"skill": skill, "version": version, "args": args, "input": input}
    if isinstance(revision, dict):
        commit = revision.get("commit")
        if isinstance(commit, str) and commit:
            entry["commit"] = commit
        repo = revision.get("repo")
        if isinstance(repo, str) and repo:
            entry["repo"] = repo
        if revision.get("dirty") is True:
            entry["dirty"] = True
    return entry


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
