#!/usr/bin/env python3
"""Ferrite README / GitHub social-preview banner, 1280x640 (2:1)."""
import sys
from gfxlib import DEFS, background, strands, toroid, glossy

W, H = 1280, 640
o = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">', DEFS]
o.append(background(W, H, [(900, 260, 520, 300, "glowTop"), (360, 560, 520, 160, "glowForge"), (1150, 420, 300, 300, "glowMag")]))
o.append(strands([((-40, -10), (40, 600), (700, 800), (200, 320)), ((1290, 1330), (-40, 600), (980, 1080), (200, 320))],
                 seed=11, n=18, sparks=140, W=W, H=H))
o.append(toroid(885, 268, R=210, r=78, SC=0.86))
# wordmark
o.append(glossy(78, 300, "ferrite", 168, family="Avenir Next", weight=900, anchor="start", spacing=2))
o.append('<text x="84" y="372" font-family="Avenir Next" font-weight="800" font-size="36" fill="#ffffff" letter-spacing="1">'
         'A Rust inference engine. Down to the GPU kernels.</text>')
o.append('<text x="86" y="418" font-family="Avenir Next" font-weight="600" font-size="24" fill="#ffd9b8" letter-spacing="4">'
         'QWEN3.8-27B  ·  NVIDIA DGX SPARK  ·  ONE OR TWO NODES</text>')
# stat chips
chips = [("473 tok/s", "code · 4 streams"), ("582 tok/s", "code edits"), ("128K", "context"), ("KL 0.021", "vs BF16")]
x = 84
for big, small in chips:
    w = 238
    o.append(f'<rect x="{x}" y="470" width="{w}" height="96" rx="20" fill="#160a07" stroke="url(#frame)" stroke-width="2.5"/>')
    o.append(f'<rect x="{x}" y="470" width="{w}" height="96" rx="20" fill="none" stroke="url(#frame)" stroke-width="6" opacity="0.5" filter="url(#blur10)"/>')
    o.append(f'<text x="{x + w / 2}" y="514" font-family="Avenir Next Condensed" font-weight="900" font-size="40" fill="#ffd27a" text-anchor="middle">{big}</text>')
    o.append(f'<text x="{x + w / 2}" y="547" font-family="Avenir Next" font-weight="600" font-size="20" fill="#e9d6cc" text-anchor="middle">{small}</text>')
    x += w + 22
o.append('<rect x="0" y="626" width="1280" height="14" fill="url(#moltenH)" opacity="0.9" filter="url(#glowBig)"/>')
o.append('</svg>')
from gfxlib import finish
sys.stdout.write(finish("".join(o)))
