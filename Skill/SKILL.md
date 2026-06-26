---
name: gemini-watermark-remover
description: >-
  Remove the Gemini / Veo "spark" watermark from a 16:9 video with no visible
  trace, preserving the scene and audio. Use when the user wants to remove the
  Gemini/Veo sparkle watermark from an AI-generated clip they own, says "remove
  the Gemini watermark", "clean this Veo video", "take off the spark/logo", or
  hands over a 16:9 Veo/Gemini MP4 to de-watermark. Tuned for 16:9 landscape
  (watermark bottom-right, size ~0.0667x frame height). Auto-detects the spark,
  then removes it per-frame with ML inpainting (LaMa) for static/structural
  scenes or inverse alpha-compositing for moving backgrounds.
license: For the user's own AI-generated content. Does not affect invisible provenance (e.g. SynthID).
---

# Gemini / Veo Watermark Remover (16:9)

Removes the visible Gemini "spark"/sparkle watermark from a video so the result
is indistinguishable from an unwatermarked clip — the scene (objects, texture,
motion) and the audio are preserved.

## When to use
- The user has a **16:9** Veo/Gemini clip **they generated/own** and wants the
  visible spark watermark gone.
- Triggers: "remove the Gemini watermark", "clean this Veo video", "de-watermark
  this", "take the spark off", or they drop in a 16:9 Veo MP4.

Scope note: this removes the **visible branding mark** only. It does not target
invisible provenance (e.g. SynthID), which is out of scope and left intact. Use
on content the user has the right to edit.

## How it works (verified pipeline)
1. **Extract** frames + audio with ffmpeg.
2. **Auto-detect** the watermark with a *removal-residual* detector on the
   temporal-mean frame: it searches the bottom-right (where Veo stamps 16:9
   clips) for the spot+size where inverse-compositing the spark template best
   *flattens* the image. Resolution-adaptive — the spark scales with height
   (~48px@720p, 72px@1080p, 144px@4K).
3. **Calibrate opacity (α)** per clip via an opacity zero-crossing (Veo's spark
   is ~30% white; calibration locks the exact value).
4. **Remove per frame** with the right engine:
   - **LaMa ML inpainting** (default for *static/structural* scenes) — rebuilds
     real structure, including objects that cross the watermark (e.g. a pen).
     Run **per frame** so reconstructed structure reconnects to each frame
     (avoids the "broken object" artifact a single shared fill causes when the
     scene drifts).
   - **Inverse alpha-compositing** (default for *moving* backgrounds) — the
     watermark is semi-transparent, so the true background is recoverable
     frame-by-frame; instant and clean where content is in motion.
   - Engine is auto-chosen from the region's temporal variance (low→LaMa,
     high→composite); override with `--engine`.
5. **Re-encode** (CRF 15) and **remux the original audio**.

## Usage
```bash
# one-time setup (ffmpeg + python deps; downloads the LaMa model on first run)
bash "scripts/setup.sh"

# remove the watermark
python3 "scripts/remove_watermark.py" INPUT.mp4 -o OUTPUT_clean.mp4

# options
#   --engine auto|lama|composite   (default: auto)
#   --mask   assets/gemini_watermark.png   (the spark template; default included)
#   --pos    "cx,cy,size"          (skip detection; force the watermark box)
#   --crf    15                    (output quality; lower = higher quality)
#   --keep-temp                    (keep extracted frames for inspection)
```

## Agent instructions (how Claude should run this)
1. Confirm the input is **16:9**; if not, warn that detection is tuned for 16:9
   and offer `--pos` to set the box manually.
2. Ensure deps: run `scripts/setup.sh` (needs `ffmpeg` + `pip install -r
   requirements.txt`). LaMa downloads `big-lama.pt` (~196 MB) to the torch cache
   on first run.
3. Run `scripts/remove_watermark.py` on the clip. LaMa mode is CPU and takes
   ~2–3 s/frame (~3 min for a 4 s clip); composite mode is near-instant.
4. **Verify** before declaring done: extract a few output frames (start, middle,
   end) and confirm the spark is gone and any object crossing it (pen, edge) is
   continuous. The previously-known failure mode is the *first/last* frames when
   the scene drifts — per-frame LaMa fixes this, so check the ends specifically.
5. Report the output path, engine used, detected position/size, and α.

## Quality bar (what "done" means)
Verified on real Veo 720p and 1080p clips: residual watermark signal at the
**measurement noise floor** (no detectable spark), **no temporal flicker**
(matches the source), **grain-matched** to the surroundings, structure (e.g.
pen) reconstructed continuously across all frames, audio preserved. See
`examples/before_after_example.png`.

## Limitations / honest notes
- Tuned for **16:9** (bottom-right placement). Other aspect ratios may need
  `--pos`.
- **LaMa is CPU-bound here** (minutes per clip). For long clips prefer
  `--engine composite` if the background is in motion (it's clean there and
  instant); use `lama` for static scenes / objects crossing the mark.
- Assumes a **single** spark watermark in the bottom-right. Multiple/other
  positions: set `--pos` or extend the detector's search region.
- Removes the **visible** mark only; invisible provenance is unaffected.

## Files
- `scripts/remove_watermark.py` — the engine (detection + LaMa/composite + mux).
- `scripts/setup.sh` — installs ffmpeg note + python deps.
- `assets/gemini_watermark.png` — the 48×48 spark template (alpha shape).
- `requirements.txt` — Python dependencies.
- `examples/before_after_example.png` — verified before/after proof.
- `README.md` — quickstart + method writeup.
