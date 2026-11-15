# Graphics

The images in `docs/assets/` are generated SVGs, rendered to PNG with librsvg: `rsvg-convert`, or `render.py` (the
same library through Python's `gi` bindings, for machines without the `rsvg-convert` binary):

```bash
python3 hero.py > ferrite-hero.svg     && python3 render.py ferrite-hero.svg ferrite-hero.png 2560
python3 banner.py > ferrite-banner.svg && python3 render.py ferrite-banner.svg ferrite-banner.png 2560
python3 render.py ferrite-banner.svg ferrite-social-preview-1280x640.png 1280
python3 bench.py bench-data.json > ferrite-bench.svg && python3 render.py ferrite-bench.svg ferrite-bench.png 2560
```

Font: [Inter](https://github.com/rsms/inter) (SIL Open Font License), the static TTFs of its release
(`extras/ttf/`, Inter and Inter Display) in `~/.local/share/fonts`. Use the static files, not the variable font: the
glossy titles stroke the glyph outlines, and the variable font's overlapping contours show up as stray lines.

`bench.py` plots only measured points from `bench-data.json`; a missing measurement is drawn as a dashed guide, never
interpolated. `tools/bench/sweep.sh` (or `sweep.py`) writes that file from a measured sweep: a 3 x 2 grid
(`"grid"`), panels with their own context points (`"contexts"`) and a text tile (`"type": "text"`). Its raw runs and
summary go to `docs/bench-data/`.
