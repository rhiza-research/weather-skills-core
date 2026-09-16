"""Write Plotly figures to PNG/HTML and dump the resolved plot spec sidecar."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from weather_skills_core.plot_spec import dump_spec, sidecar_path
from weather_skills_core.plot_style import DEFAULT_DPI


def export_png(fig, path, *, width=None, height=None, scale=1) -> Path:
    """Write a PNG via Kaleido. Width/height come from the figure layout if omitted."""
    import plotly.io as pio

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    layout = fig.layout
    width = width or layout.width or int(8 * DEFAULT_DPI)
    height = height or layout.height or int(5 * DEFAULT_DPI)
    pio.write_image(
        fig, str(output), format="png", width=int(width), height=int(height), scale=scale
    )
    return output


def export_html(fig, path, *, include_plotlyjs="cdn") -> Path:
    """Write a standalone interactive HTML figure."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output), include_plotlyjs=include_plotlyjs, full_html=True)
    return output


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
        out["upstream_history"] = raw
        break
    return out


def write_plot_outputs(
    fig,
    resolved_spec: dict,
    output: Path,
    *,
    datasets=None,
    dump_spec_path=None,
    html_path=None,
) -> Path:
    """Write ``--output`` (png or html) plus the resolved ``*.plot.json`` sidecar."""
    output = Path(output)
    suffix = output.suffix.lower()
    if suffix in {".html", ".htm"}:
        export_html(fig, output)
    else:
        export_png(fig, output)
        if html_path:
            export_html(fig, html_path)

    spec_out = dict(resolved_spec)
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
