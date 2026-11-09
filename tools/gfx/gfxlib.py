"""Shared pieces for the Ferrite graphics: molten-rust palette, SVG defs, background, ember strands, toroid, glossy text."""
import math
import random

PAL = ["#fff4d6", "#ffd27a", "#ff9a3c", "#ff6a1a", "#e8401c", "#b0201f", "#7d1238"]


def lerp(a, b, t):
    return a + (b - a) * t


def mix(c1, c2, t):
    c1 = [int(c1[i:i + 2], 16) for i in (1, 3, 5)]
    c2 = [int(c2[i:i + 2], 16) for i in (1, 3, 5)]
    return "#%02x%02x%02x" % tuple(int(lerp(a, b, t)) for a, b in zip(c1, c2))


def pal(t):
    t = max(0.0, min(1.0, t)) * (len(PAL) - 1)
    i = min(int(t), len(PAL) - 2)
    return mix(PAL[i], PAL[i + 1], t - i)


DEFS = '''<defs>
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
<linearGradient id="bar" x1="0" y1="0" x2="1" y2="0">
  <stop offset="0" stop-color="#8e1b12"/><stop offset="0.5" stop-color="#ff6a1a"/><stop offset="1" stop-color="#ffd27a"/></linearGradient>
<linearGradient id="barTF" x1="0" y1="0" x2="1" y2="0">
  <stop offset="0" stop-color="#3a4a66"/><stop offset="1" stop-color="#8fa3c4"/></linearGradient>
<linearGradient id="floor" x1="0" y1="0" x2="0" y2="1">
  <stop offset="0" stop-color="#ff6a1a" stop-opacity="0.35"/><stop offset="1" stop-color="#000" stop-opacity="0"/></linearGradient>
<filter id="blur4" x="-50%" y="-50%" width="200%" height="200%"><feGaussianBlur stdDeviation="4"/></filter>
<filter id="blur10" x="-50%" y="-50%" width="200%" height="200%"><feGaussianBlur stdDeviation="10"/></filter>
<filter id="blur24" x="-50%" y="-50%" width="200%" height="200%"><feGaussianBlur stdDeviation="24"/></filter>
<filter id="glow" x="-50%" y="-50%" width="200%" height="200%">
  <feGaussianBlur stdDeviation="3" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
<filter id="glowBig" x="-50%" y="-50%" width="200%" height="200%">
  <feGaussianBlur stdDeviation="8" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
</defs>'''


def background(W, H, glows):
    out = [f'<rect width="{W}" height="{H}" fill="url(#bg)"/>']
    for cx, cy, rx, ry, gid in glows:
        out.append(f'<ellipse cx="{cx}" cy="{cy}" rx="{rx}" ry="{ry}" fill="url(#{gid})"/>')
    hx, s = [], 46
    for row in range(-1, int(H / (s * 1.7)) + 2):
        for col in range(-1, int(W / (s * 1.5)) + 2):
            cx = col * s * 1.5
            cy = row * s * math.sqrt(3) + (col % 2) * s * math.sqrt(3) / 2
            pts = " ".join(f"{cx + s * 0.5 * math.cos(math.pi / 3 * k):.1f},{cy + s * 0.5 * math.sin(math.pi / 3 * k):.1f}"
                           for k in range(6))
            hx.append(f'<polygon points="{pts}"/>')
    out.append('<g fill="none" stroke="#ff6a1a" stroke-opacity="0.035" stroke-width="1">' + "".join(hx) + "</g>")
    return "".join(out)


def _bez(p, t):
    x0, y0, x1, y1, x2, y2, x3, y3 = p
    u = 1 - t
    return (u ** 3 * x0 + 3 * u * u * t * x1 + 3 * u * t * t * x2 + t ** 3 * x3,
            u ** 3 * y0 + 3 * u * u * t * y1 + 3 * u * t * t * y2 + t ** 3 * y3)


def strands(specs, seed=7, n=26, sparks=None, W=1920, H=1080, avoid=None):
    """specs: list of (x0 range, y0 range, x3 range, y3 range); ember strands with glowing beads."""
    rnd = random.Random(seed)
    lines, dots, halos = [], [], []
    for si, (x0r, y0r, x3r, y3r) in enumerate(specs):
        for k in range(n):
            x0, y0 = rnd.uniform(*x0r), rnd.uniform(*y0r)
            x3, y3 = rnd.uniform(*x3r), rnd.uniform(*y3r)
            bend = rnd.uniform(-160, 160)
            p = (x0, y0, lerp(x0, x3, 0.33), lerp(y0, y3, 0.33) + bend * rnd.uniform(0.6, 1.4),
                 lerp(x0, x3, 0.66), lerp(y0, y3, 0.66) - bend * rnd.uniform(0.4, 1.2), x3, y3)
            col = pal(rnd.uniform(0.1, 0.95))
            lines.append(f'<path d="M{p[0]:.1f},{p[1]:.1f} C{p[2]:.1f},{p[3]:.1f} {p[4]:.1f},{p[5]:.1f} {p[6]:.1f},{p[7]:.1f}" '
                         f'stroke="{col}" stroke-opacity="{rnd.uniform(0.18, 0.5):.2f}" stroke-width="{rnd.uniform(0.6, 1.6):.2f}"/>')
            for _ in range(rnd.randint(5, 11)):
                x, y = _bez(p, rnd.random())
                rr = rnd.uniform(1.2, 3.6)
                c = pal(rnd.uniform(0.0, 0.7))
                dots.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{rr:.1f}" fill="{c}"/>')
                if rr > 2.6:
                    halos.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{rr * 4:.1f}" fill="{c}" opacity="0.45"/>')
    out = ['<g fill="none">' + "".join(lines) + "</g>",
           '<g filter="url(#blur10)">' + "".join(halos) + "</g>",
           '<g filter="url(#glow)">' + "".join(dots) + "</g>"]
    if sparks:
        sp = []
        for _ in range(sparks):
            x, y = rnd.uniform(0, W), rnd.uniform(0, H)
            if avoid and avoid[0] < x < avoid[2] and avoid[1] < y < avoid[3]:
                continue
            sp.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{rnd.uniform(0.5, 1.8):.1f}" fill="{pal(rnd.uniform(0, 0.6))}" '
                      f'opacity="{rnd.uniform(0.25, 0.9):.2f}"/>')
        out.append('<g filter="url(#glow)">' + "".join(sp) + "</g>")
    return "".join(out)


def toroid(CX, CY, R=210, r=78, SC=1.0, NU=64, NV=18, tilt=64, yaw=-18, flux=True):
    ax, ay = math.radians(tilt), math.radians(yaw)

    def proj(u, v):
        x = (R + r * math.cos(v)) * math.cos(u)
        y = (R + r * math.cos(v)) * math.sin(u)
        z = r * math.sin(v)
        y, z = y * math.cos(ax) - z * math.sin(ax), y * math.sin(ax) + z * math.cos(ax)
        x, z = x * math.cos(ay) + z * math.sin(ay), -x * math.sin(ay) + z * math.cos(ay)
        return CX + x * SC, CY + y * SC * 0.92, z

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
                                          f'stroke-opacity="{0.12 + 0.55 * depth:.2f}" stroke-width="{(0.6 + 1.1 * depth) * SC:.2f}"/>'))
            verts.append((z, f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{(0.8 + 2.2 * depth) * SC:.1f}" '
                             f'fill="{mix(col, "#fff4d6", 0.35 * depth)}" opacity="{0.25 + 0.75 * depth:.2f}"/>'))
    segs.sort(key=lambda t: t[0])
    verts.sort(key=lambda t: t[0])
    out = [f'<ellipse cx="{CX}" cy="{CY}" rx="{330 * SC:.0f}" ry="{170 * SC:.0f}" fill="#ff6a1a" opacity="0.10" filter="url(#blur24)"/>']
    if flux:
        fl = []
        for k in range(7):
            rx, ry = (120 + 42 * k) * SC, (300 + 26 * k) * SC
            fl.append(f'<ellipse cx="{CX + 8 * SC:.0f}" cy="{CY + 10 * SC:.0f}" rx="{rx:.0f}" ry="{ry * 0.62:.0f}" '
                      f'transform="rotate(-18 {CX} {CY})" stroke="{pal(0.15 + 0.1 * k)}" stroke-opacity="{0.22 - 0.022 * k:.3f}" '
                      f'stroke-width="{1.6 - 0.12 * k:.2f}" stroke-dasharray="{2 + k} {7 + 2 * k}"/>')
        out.append('<g fill="none">' + "".join(fl) + "</g>")
    out.append('<g>' + "".join(t[1] for t in segs) + "</g>")
    out.append('<g filter="url(#glow)">' + "".join(t[1] for t in verts) + "</g>")
    return "".join(out)


def glossy(x, y, txt, size, family="Avenir Next", weight=900, anchor="middle", spacing=0):
    a = (f'x="{x}" y="{y}" font-family="{family}" font-weight="{weight}" font-size="{size}" '
         f'text-anchor="{anchor}" letter-spacing="{spacing}"')
    return (f'<text {a} fill="#ff6a1a" opacity="0.85" filter="url(#blur24)">{txt}</text>'
            f'<text {a} fill="#ff9a3c" opacity="0.7" filter="url(#blur10)">{txt}</text>'
            f'<text {a} fill="url(#molten)" stroke="#ffe7b0" stroke-width="{max(1.5, size / 80):.1f}" stroke-opacity="0.85">{txt}</text>'
            f'<text {a} fill="url(#spec)" opacity="0.55">{txt}</text>')


def logo(x, y, size=46):
    k = size / 46
    return (f'<g transform="translate({x},{y}) scale({k:.3f})">'
            '<circle cx="22" cy="22" r="18" fill="none" stroke="url(#moltenH)" stroke-width="7" filter="url(#glow)"/>'
            '<circle cx="22" cy="22" r="6" fill="#ffd27a" opacity="0.9"/>'
            '<text x="56" y="38" font-family="Avenir Next" font-weight="800" font-size="46" fill="#fff4ea" letter-spacing="1">ferrite</text>'
            '</g>')


def finish(svg):
    """XML declaration + every non-ASCII char as a numeric entity, so any viewer shows the same glyphs."""
    body = "".join(ch if ord(ch) < 128 else f"&#{ord(ch)};" for ch in svg)
    return '<?xml version="1.0" encoding="UTF-8"?>\n' + body
