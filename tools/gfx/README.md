# Graphics

The images in `docs/assets/` are generated SVGs, rendered to PNG with `rsvg-convert`:

```bash
python3 hero.py > ferrite-hero.svg     && rsvg-convert -w 2560 ferrite-hero.svg -o ferrite-hero.png
python3 banner.py > ferrite-banner.svg && rsvg-convert -w 2560 ferrite-banner.svg -o ferrite-banner.png
python3 bench.py bench-data.json > ferrite-bench.svg && rsvg-convert -w 2560 ferrite-bench.svg -o ferrite-bench.png
```

`bench.py` plots only measured points from `bench-data.json`; a missing measurement is drawn as a dashed guide, never
interpolated. `tools/bench/sweep.sh` (or `sweep.py`) writes that file from a measured sweep: a 3 x 2 grid
(`"grid"`), panels with their own context points (`"contexts"`) and a text tile (`"type": "text"`). Its raw runs and
summary go to `docs/bench-data/`. Fonts: Avenir Next (macOS) or any heavy sans-serif fontconfig can substitute.
