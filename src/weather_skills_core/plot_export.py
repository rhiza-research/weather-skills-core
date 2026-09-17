"""Write matplotlib figures to PNG and dump the resolved plot spec sidecar."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from weather_skills_core.errors import UsageError
from weather_skills_core.figure import DEFAULT_DPI, save_figure
from weather_skills_core.plot_mpl import attach_figure_spec
from weather_skills_core.plot_spec import dump_spec, sidecar_path


def export_png(fig, path, *, scale=1, tight=True) -> Path:
    """Write a PNG via matplotlib Agg. Pass ``tight=False`` to keep ``--figsize``."""
    return save_figure(fig, path, tight=tight, dpi=DEFAULT_DPI * float(scale or 1))


def attach_upstream_history(spec: dict, datasets: dict) -> dict:
    """Copy the first input Zarr's history onto the spec (plot step is stamped on PNG)."""
    out = dict(spec)
    for ds in datasets.values():
        raw = getattr(ds, "attrs", {}).get("weather_skills_history")
        if not raw:
            continue
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except json.JSONDecodeError:
                continue
        out["weather_skills_history"] = raw
        break
    return out


def write_plot_outputs(
    fig,
    resolved_spec: dict,
    output: Path,
    *,
    datasets=None,
    dump_spec_path=None,
    spec=None,
) -> Path:
    """Write ``--output`` PNG plus the resolved ``*.plot.json`` sidecar."""
    output = Path(output)
    suffix = output.suffix.lower()
    if suffix in {".html", ".htm"}:
        raise UsageError("plot output is PNG; HTML export is not supported")
    layout = resolved_spec.get("layout") or {}
    tight = getattr(fig, "_ws_tight", layout.get("autosize", True) is not False)
    if layout.get("figsize"):
        tight = False
    export_png(fig, output, tight=tight)

    spec_out = attach_figure_spec(resolved_spec, spec)
    if datasets:
        spec_out = attach_upstream_history(spec_out, datasets)
    spec_path = dump_spec_path
    if spec_path is None:
        spec_path = sidecar_path(output)
    if spec_path is not False:
        text = dump_spec(spec_out, None if str(spec_path) == "-" else spec_path)
        if str(spec_path) == "-":
            sys.stdout.write(text)
        else:
            print(f"Wrote spec: {spec_path}", file=sys.stderr)
    return output
