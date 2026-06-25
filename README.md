# Gemini Watermark Remover

Remove the visible **Gemini / Veo "spark" watermark** from a **16:9** video with
no visible trace — preserving the scene (objects, texture, motion) and the audio.

Built and verified on real Veo clips at 1280×720 and 1920×1080.

> Scope: removes the **visible branding mark** only, for content you own / have
> the right to edit. It does not target invisible provenance (e.g. SynthID),
> which is left intact.

---

## Two ways to use it

### 1. `skill/` — agent skill + Python CLI (best quality, traceless)
A self-contained skill (`SKILL.md` + scripts + the spark template). The engine
auto-detects the watermark and removes it per-frame:

- **LaMa ML inpainting** (static / structural scenes) — rebuilds real structure
  that crosses the mark (e.g. a pen) and reconnects it on every frame.
- **Inverse alpha-compositing** (moving backgrounds) — recovers the true
  semi-transparent background frame-by-frame; instant.

```bash
cd skill
bash scripts/setup.sh                                  # deps + LaMa model (~196 MB, once)
python3 scripts/remove_watermark.py INPUT.mp4 -o OUTPUT_clean.mp4
#   --engine auto|lama|composite   --pos "cx,cy,size"   --crf 15
```
See [`skill/README.md`](skill/README.md) and [`skill/SKILL.md`](skill/SKILL.md).

### 2. `standalone-tool/` — in-browser version (no install)
A single self-contained HTML page that runs entirely in the browser
(WebCodecs + MP4Box + mp4-muxer): auto-detects the spark and removes it with
inverse alpha-compositing. Serve the folder and open it in Chrome/Edge:

```bash
cd standalone-tool
python3 -m http.server 8000      # then open http://localhost:8000/watermark-remover.html
```

---

## How detection works
A *removal-residual* search on the temporal-mean frame finds the spot+size in the
bottom-right (16:9) where inverse-compositing the spark template best *flattens*
the image. It's resolution-adaptive — the spark scales with frame height
(~48px @720p, 72px @1080p, 144px @4K).

## Verified quality
Residual watermark signal at the measurement noise floor (no detectable spark),
no temporal flicker, grain-matched to surroundings, structure reconstructed
continuously across all frames, audio preserved.

## Limitations
- Tuned for **16:9** (bottom-right placement); other ratios may need `--pos`.
- LaMa mode is CPU here (~2–3 s/frame); use `composite` for moving backgrounds.
- Assumes a single spark in the bottom-right.

## Credits / licenses
- `standalone-tool/` bundles **MP4Box.js** and **mp4-muxer** (their respective
  open-source licenses apply).
- The skill's ML mode uses **LaMa** via `simple-lama-inpainting`.
