#!/usr/bin/env bash
# Rasterise the hero SVGs at 2x with headless Chrome, so the README PNG keeps
# the intended macOS typography instead of a viewer-side font fallback.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HERO="$ROOT/artifacts/hero"
CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
WIDTH=1680
HEIGHT=628

[ -x "$CHROME" ] || { echo "Chrome not found at $CHROME" >&2; exit 1; }

for theme in light dark; do
  svg="$HERO/hero_${theme}.svg"
  html="$HERO/.render_${theme}.html"
  printf '<!doctype html><meta charset="utf-8"><style>html,body{margin:0;padding:0;overflow:hidden}svg{display:block}</style>%s' \
    "$(cat "$svg")" > "$html"

  "$CHROME" --headless=new --disable-gpu --hide-scrollbars \
    --force-device-scale-factor=2 \
    --window-size="${WIDTH},${HEIGHT}" \
    --screenshot="$HERO/hero_${theme}.png" \
    "file://$html" >/dev/null 2>&1

  rm -f "$html"
  echo "wrote artifacts/hero/hero_${theme}.png"
done
