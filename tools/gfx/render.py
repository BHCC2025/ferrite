#!/usr/bin/env python3
"""Render a generated SVG to PNG with librsvg (the same renderer as rsvg-convert), for machines without the
rsvg-convert binary: python3 render.py IN.svg OUT.png WIDTH. Needs python3-gi with Rsvg 2.0 and pycairo."""
import sys
import gi
gi.require_version("Rsvg", "2.0")
from gi.repository import Rsvg
import cairo

src, dst, width = sys.argv[1], sys.argv[2], int(sys.argv[3])
h = Rsvg.Handle.new_from_file(src)
ok, w0, h0 = h.get_intrinsic_size_in_pixels()
if not ok:
    raise SystemExit(f"{src}: no intrinsic size")
scale = width / w0
surf = cairo.ImageSurface(cairo.FORMAT_RGB24, width, round(h0 * scale))
ctx = cairo.Context(surf)
vp = Rsvg.Rectangle(); vp.x, vp.y, vp.width, vp.height = 0, 0, width, round(h0 * scale)
h.render_document(ctx, vp)
surf.write_to_png(dst)
print(f"{dst}: {width} x {round(h0 * scale)}")
