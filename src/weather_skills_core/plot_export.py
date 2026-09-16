"""Write Plotly figures to PNG/HTML and dump the resolved plot spec sidecar."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from weather_skills_core.figure import DEFAULT_DPI, save_figure
from weather_skills_core.plot_mpl import plotly_to_mpl
from weather_skills_core.plot_spec import dump_spec, sidecar_path


def export_png(fig, path, *, width=None, height=None, scale=1, tight=True) -> Path:
    """Write a PNG via matplotlib Agg. Width/height come from the figure layout if omitted.

    Plotly/Kaleido PNG export needs Chrome; this rasterizes the traces we emit instead.
    Pass ``tight=False`` to keep an explicit ``--figsize``.
    """
    mpl_fig = plotly_to_mpl(fig, width=width, height=height)
    return save_figure(mpl_fig, path, tight=tight, dpi=DEFAULT_DPI * float(scale or 1))


def export_html(fig, path, *, include_plotlyjs="cdn") -> Path:
    """Write a standalone interactive HTML figure."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output), include_plotlyjs=include_plotlyjs, full_html=True)
    return output


def export_plotly_json(fig, path, *, include_data=False) -> Path:
    """Write Plotly figure JSON. Default is layout-only (heatmap ``z`` is huge)."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if include_data:
        output.write_text(fig.to_json(), encoding="utf-8")
        return output
    payload = fig.to_dict()
    layout_only = {
        "layout": payload.get("layout") or {},
        "data": [
            {k: v for k, v in trace.items() if k not in {"z", "y", "x"} or k == "type"}
            for trace in (payload.get("data") or [])
        ],
    }
    output.write_text(json.dumps(layout_only, indent=2, default=str) + "\n", encoding="utf-8")
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
    html_path=None,
    plotly_json_path=None,
    include_plotly_data=False,
) -> Path:
    """Write ``--output`` (png or html) plus the resolved ``*.plot.json`` sidecar."""
    output = Path(output)
    suffix = output.suffix.lower()
    if suffix in {".html", ".htm"}:
        export_html(fig, output)
    else:
        layout = getattr(fig, "layout", None)
        autosize = True if layout is None else layout.autosize is not False
        export_png(fig, output, tight=autosize)
        if html_path:
            export_html(fig, html_path)
    if plotly_json_path:
        export_plotly_json(fig, plotly_json_path, include_data=include_plotly_data)

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
