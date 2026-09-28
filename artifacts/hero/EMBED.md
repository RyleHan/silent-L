# Hero figure

Regenerate after any data change:

```bash
python3 scripts/build_hero_figure.py && bash scripts/render_hero_png.sh
```

`build_hero_figure.py` reads every number from the locked artifacts, so a
scale-up run refreshes the figure without touching the layout code.

Paste at the top of `README.md`, above the title. The `<picture>` element makes
GitHub serve the dark version to viewers on a dark theme:

```html
<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="artifacts/hero/hero_dark.png">
    <img alt="Readable is not controllable: goal state is decodable and locally causal in OpenVLA and pi0.5, but single-block intervention fails closed-loop control while full 18-block pathway replay recovers the policy exactly." src="artifacts/hero/hero_light.png" width="100%">
  </picture>
</p>
```

Files:

| File | Use |
|---|---|
| `hero_light.svg` / `hero_dark.svg` | source of truth, vector, for slides and print |
| `hero_light.png` / `hero_dark.png` | 3360x1256 (2x), for the README |
