#!/usr/bin/env python3
"""Ferrite hero graphic: molten-rust neon poster, 1920x1080 SVG (render with rsvg-convert)."""
import math
import random
import sys

W, H = 1920, 1080
rnd = random.Random(7)
out = []
add = out.append


def lerp(a, b, t):
    return a + (b - a) * t


def mix(c1, c2, t):
    c1 = [int(c1[i:i + 2], 16) for i in (1, 3, 5)]
    c2 = [int(c2[i:i + 2], 16) for i in (1, 3, 5)]
    return "#%02x%02x%02x" % tuple(int(lerp(a, b, t)) for a, b in zip(c1, c2))


# molten palette: hot white -> gold -> orange -> rust -> ember red -> deep magenta
PAL = ["#fff4d6", "#ffd27a", "#ff9a3c", "#ff6a1a", "#e8401c", "#b0201f", "#7d1238"]


def pal(t):
    t = max(0.0, min(1.0, t)) * (len(PAL) - 1)
    i = min(int(t), len(PAL) - 2)
    return mix(PAL[i], PAL[i + 1], t - i)


add(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">')
add('''<defs>
<radialGradient id="bg" cx="0.36" cy="0.30" r="0.95">
  <stop offset="0" stop-color="#3a1206"/><stop offset="0.35" stop-color="#1a0704"/>
  <stop offset="0.7" stop-color="#0b0403"/><stop offset="1" stop-color="#050202"/></radialGradient>
<radialGradient id="glowTop" cx="0.5" cy="0.5" r="0.5">
  <stop offset="0" stop-color="#ff6a1a" stop-opacity="0.55"/><stop offset="0.5" stop-color="#c2300f" stop-opacity="0.18"/>
  <stop offset="1" stop-color="#000" stop-opacity="0"/></radialGradient>
<radialGradient id="glowForge" cx="0.5" cy="0.5" r="0.5">
  <stop offset="0" stop-color="#ff8a2a" stop-opacity="0.75"/><stop offset="0.45" stop-color="#e2461a" stop-opacity="0.25"/>
  <stop offset="1" stop-color="#000" stop-opacity="0"/></radialGradient>
<radialGradient id="glowMag" cx="0.5" cy="0.5" r="0.5">
  <stop offset="0" stop-color="#d81b60" stop-opacity="0.35"/><stop offset="1" stop-color="#000" stop-opacity="0"/></radialGradient>
<linearGradient id="molten" x1="0" y1="0" x2="0" y2="1">
  <stop offset="0" stop-color="#fff6dc"/><stop offset="0.22" stop-color="#ffd27a"/><stop offset="0.48" stop-color="#ff8a2a"/>
  <stop offset="0.72" stop-color="#e2461a"/><stop offset="1" stop-color="#8e1b12"/></linearGradient>
<linearGradient id="moltenH" x1="0" y1="0" x2="1" y2="0">
  <stop offset="0" stop-color="#ffb347"/><stop offset="0.35" stop-color="#ff6a1a"/><stop offset="0.7" stop-color="#e8401c"/>
  <stop offset="1" stop-color="#c2185b"/></linearGradient>
<linearGradient id="spec" x1="0" y1="0" x2="0" y2="1">
  <stop offset="0" stop-color="#ffffff" stop-opacity="0.95"/><stop offset="0.42" stop-color="#ffffff" stop-opacity="0.25"/>
  <stop offset="0.5" stop-color="#ffffff" stop-opacity="0"/></linearGradient>
<linearGradient id="frame" x1="0" y1="0" x2="1" y2="1">
  <stop offset="0" stop-color="#ffd27a"/><stop offset="0.3" stop-color="#ff6a1a"/><stop offset="0.6" stop-color="#e8401c"/>
  <stop offset="0.85" stop-color="#c2185b"/><stop offset="1" stop-color="#ff9a3c"/></linearGradient>
<linearGradient id="cardbg" x1="0" y1="0" x2="0" y2="1">
  <stop offset="0" stop-color="#1c0d0a" stop-opacity="0.92"/><stop offset="1" stop-color="#0d0605" stop-opacity="0.96"/></linearGradient>
<linearGradient id="bar" x1="0" y1="0" x2="1" y2="0">
  <stop offset="0" stop-color="#8e1b12"/><stop offset="0.5" stop-color="#ff6a1a"/><stop offset="1" stop-color="#ffd27a"/></linearGradient>
<linearGradient id="floor" x1="0" y1="0" x2="0" y2="1">
  <stop offset="0" stop-color="#ff6a1a" stop-opacity="0.35"/><stop offset="1" stop-color="#000" stop-opacity="0"/></linearGradient>
<filter id="blur4" x="-50%" y="-50%" width="200%" height="200%"><feGaussianBlur stdDeviation="4"/></filter>
<filter id="blur10" x="-50%" y="-50%" width="200%" height="200%"><feGaussianBlur stdDeviation="10"/></filter>
<filter id="blur24" x="-50%" y="-50%" width="200%" height="200%"><feGaussianBlur stdDeviation="24"/></filter>
<filter id="glow" x="-50%" y="-50%" width="200%" height="200%">
  <feGaussianBlur stdDeviation="3" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
<filter id="glowBig" x="-50%" y="-50%" width="200%" height="200%">
  <feGaussianBlur stdDeviation="8" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
</defs>''')

# background + atmosphere
add(f'<rect width="{W}" height="{H}" fill="url(#bg)"/>')
add('<ellipse cx="660" cy="230" rx="720" ry="360" fill="url(#glowTop)"/>')
add('<ellipse cx="1500" cy="560" rx="520" ry="520" fill="url(#glowMag)"/>')
add('<ellipse cx="640" cy="905" rx="760" ry="230" fill="url(#glowForge)"/>')

# faint hex lattice (spinel crystal hint)
hx = []
s = 46
for row in range(-1, 26):
    for col in range(-1, 46):
        cx = col * s * 1.5
        cy = row * s * math.sqrt(3) + (col % 2) * s * math.sqrt(3) / 2
        pts = " ".join(f"{cx + s * 0.5 * math.cos(math.pi / 3 * k):.1f},{cy + s * 0.5 * math.sin(math.pi / 3 * k):.1f}"
                       for k in range(6))
        hx.append(f'<polygon points="{pts}"/>')
add('<g fill="none" stroke="#ff6a1a" stroke-opacity="0.035" stroke-width="1">' + "".join(hx) + "</g>")


# ember strands flowing in from both edges toward the torus
def strand(x0, y0, x3, y3, bend, seed):
    r = random.Random(seed)
    x1, y1 = lerp(x0, x3, 0.33), lerp(y0, y3, 0.33) + bend * r.uniform(0.6, 1.4)
    x2, y2 = lerp(x0, x3, 0.66), lerp(y0, y3, 0.66) - bend * r.uniform(0.4, 1.2)
    return (x0, y0, x1, y1, x2, y2, x3, y3)


def bez(p, t):
    x0, y0, x1, y1, x2, y2, x3, y3 = p
    u = 1 - t
    return (u ** 3 * x0 + 3 * u * u * t * x1 + 3 * u * t * t * x2 + t ** 3 * x3,
            u ** 3 * y0 + 3 * u * u * t * y1 + 3 * u * t * t * y2 + t ** 3 * y3)


lines, dots, halos = [], [], []
for side in (0, 1):
    for k in range(26):
        if side == 0:
            p = strand(-40, rnd.uniform(40, 520), rnd.uniform(470, 620), rnd.uniform(150, 330), rnd.uniform(-160, 160), k)
        else:
            p = strand(1960, rnd.uniform(-40, 420), rnd.uniform(1060, 1210), rnd.uniform(150, 330), rnd.uniform(-160, 160), 100 + k)
        col = pal(rnd.uniform(0.1, 0.95))
        x0, y0, x1, y1, x2, y2, x3, y3 = p
        lines.append(f'<path d="M{x0:.1f},{y0:.1f} C{x1:.1f},{y1:.1f} {x2:.1f},{y2:.1f} {x3:.1f},{y3:.1f}" '
                     f'stroke="{col}" stroke-opacity="{rnd.uniform(0.18, 0.5):.2f}" stroke-width="{rnd.uniform(0.6, 1.6):.2f}"/>')
        for j in range(rnd.randint(5, 11)):
            t = rnd.random()
            x, y = bez(p, t)
            rr = rnd.uniform(1.2, 3.6)
            c = pal(rnd.uniform(0.0, 0.7))
            dots.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{rr:.1f}" fill="{c}"/>')
            if rr > 2.6:
                halos.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{rr * 4:.1f}" fill="{c}" opacity="0.45"/>')
add('<g fill="none">' + "".join(lines) + "</g>")
add('<g filter="url(#blur10)">' + "".join(halos) + "</g>")
add('<g filter="url(#glow)">' + "".join(dots) + "</g>")

# loose sparks everywhere
sp = []
for _ in range(170):
    x, y = rnd.uniform(0, W), rnd.uniform(0, H)
    if 1180 < x < 1880 and 90 < y < 1010:
        continue
    sp.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{rnd.uniform(0.5, 1.8):.1f}" fill="{pal(rnd.uniform(0, 0.6))}" '
              f'opacity="{rnd.uniform(0.25, 0.9):.2f}"/>')
add('<g filter="url(#glow)">' + "".join(sp) + "</g>")

# ---- hero: wireframe ferrite toroid core with magnetic flux loops ----
CX, CY, SC = 840, 250, 1.0
R, r = 210, 78
ax, ay = math.radians(64), math.radians(-18)


def proj(u, v):
    x = (R + r * math.cos(v)) * math.cos(u)
    y = (R + r * math.cos(v)) * math.sin(u)
    z = r * math.sin(v)
    # rotate about x then y
    y, z = y * math.cos(ax) - z * math.sin(ax), y * math.sin(ax) + z * math.cos(ax)
    x, z = x * math.cos(ay) + z * math.sin(ay), -x * math.sin(ay) + z * math.cos(ay)
    return CX + x * SC, CY + y * SC * 0.92, z


NU, NV = 64, 18
segs, verts = [], []
for i in range(NU):
    for j in range(NV):
        u, v = 2 * math.pi * i / NU, 2 * math.pi * j / NV
        u2, v2 = 2 * math.pi * (i + 1) / NU, 2 * math.pi * (j + 1) / NV
        x, y, z = proj(u, v)
        depth = (z + R + r) / (2 * (R + r))
        col = pal(0.08 + 0.85 * (0.5 + 0.5 * math.sin(u - 0.6)))
        for (xa, ya, za) in (proj(u2, v), proj(u, v2)):
            segs.append((min(z, za), f'<line x1="{x:.1f}" y1="{y:.1f}" x2="{xa:.1f}" y2="{ya:.1f}" stroke="{col}" '
                                      f'stroke-opacity="{0.12 + 0.55 * depth:.2f}" stroke-width="{0.6 + 1.1 * depth:.2f}"/>'))
        verts.append((z, f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{0.8 + 2.2 * depth:.1f}" fill="{mix(col, "#fff4d6", 0.35 * depth)}" '
                         f'opacity="{0.25 + 0.75 * depth:.2f}"/>'))
segs.sort(key=lambda t: t[0])
verts.sort(key=lambda t: t[0])
add(f'<ellipse cx="{CX}" cy="{CY}" rx="330" ry="170" fill="#ff6a1a" opacity="0.10" filter="url(#blur24)"/>')
# flux loops threading the core
flux = []
for k in range(7):
    a = 0.55 + 0.11 * k
    rx, ry = 120 + 42 * k, 300 + 26 * k
    flux.append(f'<ellipse cx="{CX + 8}" cy="{CY + 10}" rx="{rx}" ry="{ry * 0.62:.0f}" transform="rotate(-18 {CX} {CY})" '
                f'stroke="{pal(0.15 + 0.1 * k)}" stroke-opacity="{0.22 - 0.022 * k:.3f}" stroke-width="{1.6 - 0.12 * k:.2f}" '
                f'stroke-dasharray="{2 + k} {7 + 2 * k}"/>')
add('<g fill="none">' + "".join(flux) + "</g>")
add('<g>' + "".join(t[1] for t in segs) + "</g>")
add('<g filter="url(#glow)">' + "".join(t[1] for t in verts) + "</g>")

# ---- logo (top-left) ----
add('<g transform="translate(92,72)">'
    '<circle cx="22" cy="22" r="18" fill="none" stroke="url(#moltenH)" stroke-width="7" filter="url(#glow)"/>'
    '<circle cx="22" cy="22" r="6" fill="#ffd27a" opacity="0.9"/>'
    '<text x="56" y="38" font-family="Avenir Next" font-weight="800" font-size="46" fill="#fff4ea" letter-spacing="1">ferrite</text>'
    '</g>')


# ---- headline + hero number (left column) ----
def glossy(x, y, txt, size, family="Avenir Next", weight=900, anchor="middle", spacing=0):
    a = (f'x="{x}" y="{y}" font-family="{family}" font-weight="{weight}" font-size="{size}" '
         f'text-anchor="{anchor}" letter-spacing="{spacing}"')
    return (f'<text {a} fill="#ff6a1a" opacity="0.85" filter="url(#blur24)">{txt}</text>'
            f'<text {a} fill="#ff9a3c" opacity="0.7" filter="url(#blur10)">{txt}</text>'
            f'<text {a} fill="url(#molten)" stroke="#ffe7b0" stroke-width="2.5" stroke-opacity="0.85">{txt}</text>'
            f'<text {a} fill="url(#spec)" opacity="0.55">{txt}</text>')


LX = 640
add(f'<text x="{LX}" y="520" font-family="Avenir Next" font-weight="900" font-size="74" text-anchor="middle" '
    f'fill="#ffffff" letter-spacing="2" filter="url(#glow)">RUST ALL THE WAY DOWN.</text>')
add(f'<text x="{LX}" y="574" font-family="Avenir Next" font-weight="700" font-size="30" text-anchor="middle" '
    f'fill="#ffd9b8" letter-spacing="6">HOST CODE  ·  GPU KERNELS  ·  SERVER</text>')
add(glossy(LX + 10, 790, "473 tok/s", 196, family="Avenir Next Condensed", weight=900, spacing=2))
add(f'<text x="{LX}" y="866" font-family="Avenir Next" font-weight="800" font-size="40" text-anchor="middle" '
    f'fill="#ffffff" letter-spacing="3">QWEN3.8-27B  ·  4 STREAMS  ·  TWO DGX SPARKS</text>')

# molten pedestal
px, py, pw, ph = 120, 905, 1040, 46
add(f'<rect x="{px - 20}" y="{py - 30}" width="{pw + 40}" height="{ph + 70}" rx="30" fill="#ff6a1a" opacity="0.35" filter="url(#blur24)"/>')
add(f'<rect x="{px}" y="{py}" width="{pw}" height="{ph}" rx="22" fill="#2a0e08" fill-opacity="0.8" stroke="url(#frame)" stroke-width="3"/>')
add(f'<rect x="{px + 70}" y="{py + 14}" width="{pw - 140}" height="{ph - 28}" rx="9" fill="url(#moltenH)" filter="url(#glowBig)"/>')
add(f'<rect x="{px + 300}" y="{py + 17}" width="{pw - 600}" height="{ph - 34}" rx="6" fill="#fff4d6" opacity="0.85" filter="url(#blur4)"/>')
add(f'<rect x="{px}" y="{py + ph + 6}" width="{pw}" height="90" fill="url(#floor)" opacity="0.6"/>')

add(f'<text x="96" y="1032" font-family="Avenir Next" font-weight="800" font-size="34" fill="#ffffff" letter-spacing="1">'
    f'AN OPEN INFERENCE ENGINE FOR NVIDIA DGX SPARK.</text>')

# ---- right: glass benchmark card with neon rust frame ----
cx0, cy0, cw, ch = 1250, 110, 590, 880
add(f'<rect x="{cx0 - 14}" y="{cy0 - 14}" width="{cw + 28}" height="{ch + 28}" rx="64" fill="none" stroke="url(#frame)" '
    f'stroke-width="18" opacity="0.55" filter="url(#blur24)"/>')
add(f'<rect x="{cx0}" y="{cy0}" width="{cw}" height="{ch}" rx="56" fill="none" stroke="url(#frame)" stroke-width="10" opacity="0.8" filter="url(#blur10)"/>')
add(f'<rect x="{cx0}" y="{cy0}" width="{cw}" height="{ch}" rx="56" fill="#140907" stroke="url(#frame)" stroke-width="6"/>')
add(f'<rect x="{cx0 + 18}" y="{cy0 + 18}" width="{cw - 36}" height="{ch - 36}" rx="42" fill="none" stroke="#ffffff" stroke-opacity="0.08" stroke-width="2"/>')
add(f'<text x="{cx0 + 52}" y="{cy0 + 92}" font-family="Avenir Next" font-weight="800" font-size="34" fill="#fff4ea">Live on two DGX Sparks</text>')
add(f'<text x="{cx0 + 52}" y="{cy0 + 132}" font-family="Menlo" font-size="21" fill="#ffb98a">Qwen3.8-27B  EXL3 3.0 bpw  128K ctx</text>')
add(f'<line x1="{cx0 + 52}" y1="{cy0 + 162}" x2="{cx0 + cw - 52}" y2="{cy0 + 162}" stroke="#ffffff" stroke-opacity="0.12" stroke-width="2"/>')

rows = [
    ("Code edits, 1 stream", "582", "tok/s", 582),
    ("Code, 4 streams", "473", "tok/s", 473),
    ("Code, 1 stream", "273", "tok/s", 273),
    ("Code, 1 stream @32K", "238", "tok/s", 238),
    ("Prefill @2K", "3,550", "tok/s", None),
]
y = cy0 + 225
for label, val, unit, v in rows:
    add(f'<text x="{cx0 + 52}" y="{y}" font-family="Avenir Next" font-weight="600" font-size="27" fill="#e9d6cc">{label}</text>')
    add(f'<text x="{cx0 + cw - 52}" y="{y}" font-family="Avenir Next Condensed" font-weight="800" font-size="40" fill="#ffd27a" '
        f'text-anchor="end">{val}<tspan font-size="22" fill="#ff9a3c" font-family="Avenir Next" font-weight="600"> {unit}</tspan></text>')
    if v is not None:
        bw = (cw - 104) * v / 582
        add(f'<rect x="{cx0 + 52}" y="{y + 18}" width="{cw - 104}" height="10" rx="5" fill="#ffffff" opacity="0.07"/>')
        add(f'<rect x="{cx0 + 52}" y="{y + 18}" width="{bw:.0f}" height="10" rx="5" fill="url(#bar)" filter="url(#glow)"/>')
    y += 96

add(f'<line x1="{cx0 + 52}" y1="{y - 40}" x2="{cx0 + cw - 52}" y2="{y - 40}" stroke="#ffffff" stroke-opacity="0.12" stroke-width="2"/>')
add(f'<text x="{cx0 + 52}" y="{y + 18}" font-family="Avenir Next" font-weight="600" font-size="27" fill="#e9d6cc">KL vs BF16 (code)</text>')
add(f'<text x="{cx0 + cw - 52}" y="{y + 18}" font-family="Avenir Next Condensed" font-weight="800" font-size="40" fill="#ffd27a" text-anchor="end">0.021</text>')
add(f'<text x="{cx0 + 52}" y="{y + 70}" font-family="Avenir Next" font-weight="600" font-size="27" fill="#e9d6cc">Lossless speculation</text>')
add(f'<text x="{cx0 + cw - 52}" y="{y + 70}" font-family="Avenir Next" font-weight="800" font-size="27" fill="#ffd27a" text-anchor="end">DFlash2 + copy</text>')
add(f'<text x="{cx0 + cw / 2}" y="{cy0 + ch - 44}" font-family="Menlo" font-size="18" fill="#a98b80" text-anchor="middle">'
    f'greedy decoding · medians of 3-5 runs · Nov 2026</text>')

add(f'<text x="1840" y="1040" font-family="Avenir Next" font-weight="600" font-size="22" fill="#cdb2a6" text-anchor="end">'
    f'built on exllamav3 \u00b7 TensorFold \u00b7 DFlash2 \u00b7 Qwen3.8</text>')
add('</svg>')
from gfxlib import finish
sys.stdout.write(finish("\n".join(out)))
