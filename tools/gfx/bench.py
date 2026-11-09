#!/usr/bin/env python3
"""Ferrite benchmark poster, 1920x1080. Data from a JSON file (argv[1]); missing points are drawn
as 'not measured' gaps, never interpolated. An optional second series ("ref") is supported for comparing two Ferrite builds or settings.
Layout: "grid": [columns, rows] (default [2, 2]); a panel may set its own "contexts" / "ctx_labels" (default the
file's), and a panel with "type": "text" is a tile of label / value lines (tools/bench/sweep.py writes such a file)."""
import json
import math
import sys
from gfxlib import DEFS, background, strands, logo

data = json.load(open(sys.argv[1]))
W, H = 1920, 1080
FE, REF = "#e8641c", "#5a95e0"         # validated pair on the dark surface: molten orange, steel blue
INK, INK2, MUTED, GRID = "#f4ece8", "#cdbdb5", "#8f7d75", "#ffffff"
CTX = data["contexts"]                   # e.g. [2048, 8192, 32768, 65536, 122880]
o = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">', DEFS]
o.append(background(W, H, [(960, 120, 900, 220, "glowTop"), (960, 1060, 900, 160, "glowForge")]))
o.append(strands([], seed=5, n=0, sparks=90, W=W, H=150, avoid=(360, 0, 1480, 150)))

# header
o.append(logo(72, 46, 40))
o.append(f'<text x="960" y="88" font-family="Avenir Next" font-weight="900" font-size="50" fill="{INK}" text-anchor="middle" '
         f'letter-spacing="1" filter="url(#glow)">{data["title"]}</text>')
o.append(f'<text x="960" y="128" font-family="Avenir Next" font-weight="600" font-size="23" fill="{INK2}" text-anchor="middle" '
         f'letter-spacing="2">{data["subtitle"]}</text>')
# legend (always present for 2 series), top-right
lx = 1500
SERIES = [("fe", FE)] + ([("ref", REF)] if data.get("ref_name") else [])
for name, col, ly in (((data["fe_name"], FE, 62), (data["ref_name"], REF, 98)) if data.get("ref_name") else ()):
    o.append(f'<line x1="{lx}" y1="{ly}" x2="{lx + 40}" y2="{ly}" stroke="{col}" stroke-width="3"/>')
    o.append(f'<circle cx="{lx + 20}" cy="{ly}" r="6" fill="{col}" stroke="#140907" stroke-width="2"/>')
    o.append(f'<text x="{lx + 54}" y="{ly + 8}" font-family="Avenir Next" font-weight="700" font-size="22" fill="{INK}">{name}</text>')


def xpos(c, x0, w, ctx=CTX):
    lo, hi = math.log(ctx[0]), math.log(ctx[-1])
    return x0 + 40 + (math.log(c) - lo) / (hi - lo) * (w - 80)


def fmt(v):
    return f"{v:,.0f}"


def text_tile(px, py, pw, ph, p):
    out = [f'<rect x="{px}" y="{py}" width="{pw}" height="{ph}" rx="28" fill="#140907" stroke="url(#frame)" stroke-width="2"/>',
           f'<text x="{px + 34}" y="{py + 50}" font-family="Avenir Next" font-weight="800" font-size="28" fill="{INK}">{p["title"]}</text>',
           f'<text x="{px + 34}" y="{py + 80}" font-family="Avenir Next" font-weight="600" font-size="18" fill="{MUTED}">{p["unit"]}</text>']
    for j, (lab, val) in enumerate(p["lines"]):
        yy = py + 150 + j * 92
        out.append(f'<text x="{px + 34}" y="{yy}" font-family="Avenir Next" font-weight="700" font-size="20" fill="{INK2}">{lab}</text>')
        out.append(f'<text x="{px + 34}" y="{yy + 46}" font-family="Avenir Next Condensed" font-weight="800" font-size="44" '
                   f'fill="{FE}" filter="url(#glow)">{val}</text>')
    if p.get("foot"):
        out.append(f'<text x="{px + 34}" y="{py + ph - 30}" font-family="Avenir Next" font-weight="600" font-size="17" fill="{MUTED}">{p["foot"]}</text>')
    return "".join(out)


def panel(px, py, pw, ph, p):
    if p.get("type") == "text":
        return text_tile(px, py, pw, ph, p)
    out = []
    ctx = p.get("contexts", CTX)
    labels = p.get("ctx_labels", data["ctx_labels"])
    out.append(f'<rect x="{px}" y="{py}" width="{pw}" height="{ph}" rx="28" fill="#140907" stroke="url(#frame)" stroke-width="2"/>')
    out.append(f'<rect x="{px}" y="{py}" width="{pw}" height="{ph}" rx="28" fill="none" stroke="url(#frame)" stroke-width="5" opacity="0.35" filter="url(#blur10)"/>')
    out.append(f'<text x="{px + 34}" y="{py + 50}" font-family="Avenir Next" font-weight="800" font-size="28" fill="{INK}">{p["title"]}</text>')
    out.append(f'<text x="{px + 34}" y="{py + 80}" font-family="Avenir Next" font-weight="600" font-size="18" fill="{MUTED}">{p["unit"]}</text>')
    # plot area
    x0, y0, w, h = px + 100, py + 112, pw - 150, ph - 180
    vals = [v for s, _ in SERIES for pt in p[s] if pt for v in (pt[0], pt[2] if len(pt) > 2 else pt[0])]
    ymax = p.get("ymax") or max(vals) * 1.18
    step = p["ystep"]
    yv = 0
    while yv <= ymax + 1e-9:
        yy = y0 + h - yv / ymax * h
        out.append(f'<line x1="{x0}" y1="{yy:.1f}" x2="{x0 + w}" y2="{yy:.1f}" stroke="{GRID}" stroke-opacity="{0.18 if yv == 0 else 0.07}" stroke-width="1"/>')
        out.append(f'<text x="{x0 - 14}" y="{yy + 6:.1f}" font-family="Avenir Next" font-weight="600" font-size="17" fill="{MUTED}" text-anchor="end">{fmt(yv)}</text>')
        yv += step
    for c, lab in zip(ctx, labels):
        xx = xpos(c, x0, w, ctx)
        out.append(f'<text x="{xx:.1f}" y="{y0 + h + 32}" font-family="Avenir Next" font-weight="700" font-size="18" fill="{INK2}" text-anchor="middle">{lab}</text>')
    out.append(f'<text x="{x0 + w / 2}" y="{y0 + h + 58}" font-family="Avenir Next" font-weight="600" font-size="16" fill="{MUTED}" text-anchor="middle">context length (tokens)</text>')

    def Y(v):
        return y0 + h - v / ymax * h

    for i, c in enumerate(ctx):
        fe_m = p["fe"][i] is None
        tf_m = p["ref"][i] is None if "ref" in p else False
        if fe_m or tf_m:
            xx = xpos(c, x0, w, ctx)
            cap = "not measured yet" if (fe_m and tf_m) else (("Ferrite" if fe_m else "reference") + " not measured")
            out.append(f'<line x1="{xx:.1f}" y1="{y0 + 8}" x2="{xx:.1f}" y2="{y0 + h}" stroke="{MUTED}" stroke-opacity="0.35" stroke-width="1.5" stroke-dasharray="4 6"/>')
    for key, col, dy in [(k, c, -1 if k == "fe" else 1) for k, c in reversed(SERIES)]:
        pts = [(xpos(c, x0, w, ctx), pt) for c, pt in zip(ctx, p[key])]
        # line segments only between consecutive measured points (gaps stay gaps)
        for (xa, a), (xb, b) in zip(pts, pts[1:]):
            if a and b:
                out.append(f'<line x1="{xa:.1f}" y1="{Y(a[0]):.1f}" x2="{xb:.1f}" y2="{Y(b[0]):.1f}" stroke="{col}" stroke-width="3" '
                           f'stroke-linecap="round" filter="url(#glow)"/>')
        for i, (xx, pt) in enumerate(pts):
            if not pt:
                continue
            v = pt[0]
            if len(pt) > 2 and pt[1] != pt[2]:
                out.append(f'<line x1="{xx:.1f}" y1="{Y(pt[1]):.1f}" x2="{xx:.1f}" y2="{Y(pt[2]):.1f}" stroke="{col}" stroke-width="2" opacity="0.75"/>')
                for e in (pt[1], pt[2]):
                    out.append(f'<line x1="{xx - 7:.1f}" y1="{Y(e):.1f}" x2="{xx + 7:.1f}" y2="{Y(e):.1f}" stroke="{col}" stroke-width="2" opacity="0.75"/>')
            out.append(f'<circle cx="{xx:.1f}" cy="{Y(v):.1f}" r="7" fill="{col}" stroke="#140907" stroke-width="2.5"/>')
            # direct value labels (text ink, not series color), Ferrite above, reference below
            ty = Y(v) - 16 if dy < 0 else Y(v) + 30
            out.append(f'<text x="{xx:.1f}" y="{ty:.1f}" font-family="Avenir Next Condensed" font-weight="800" font-size="21" '
                       f'fill="{INK}" text-anchor="middle">{fmt(v)}</text>')
    if p.get("note"):
        out.append(f'<text x="{px + pw - 34}" y="{py + 50}" font-family="Avenir Next" font-weight="700" font-size="17" fill="{MUTED}" text-anchor="end">{p["note"]}</text>')
    return "".join(out)


panels = data["panels"]
cols, rows = data.get("grid", [2, 2])
gx, gy, gap, gh = 60, 168, 30, 400
gw = (W - 2 * gx - (cols - 1) * gap) // cols
for i, p in enumerate(panels[:cols * rows]):
    px = gx + (i % cols) * (gw + gap)
    py = gy + (i // cols) * (gh + 22)
    o.append(panel(px, py, gw, gh, p))

# footer: method + accuracy strip
fy = 1016
o.append(f'<text x="60" y="{fy}" font-family="Avenir Next" font-weight="700" font-size="19" fill="{INK2}">{data["method"]}</text>')
o.append(f'<text x="60" y="{fy + 32}" font-family="Avenir Next" font-weight="600" font-size="17" fill="{MUTED}">{data["method2"]}</text>')
o.append(f'<text x="1860" y="{fy}" font-family="Avenir Next" font-weight="800" font-size="21" fill="{INK}" text-anchor="end">{data["kl"]}</text>')
o.append(f'<text x="1860" y="{fy + 32}" font-family="Avenir Next" font-weight="600" font-size="17" fill="{MUTED}" text-anchor="end">{data["kl2"]}</text>')
o.append('</svg>')
from gfxlib import finish
sys.stdout.write(finish("".join(o)))
