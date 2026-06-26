#!/usr/bin/env python3
"""
Gemini / Veo Watermark Remover  —  optimized for 16:9 landscape clips.

Removes the Gemini "spark" watermark from a video with no visible trace,
preserving the scene and audio.

Two engines (auto-selected, or forced with --engine):
  • lama       ML inpainting per-frame (default for static/structural scenes).
               Rebuilds real structure (objects crossing the mark). Best quality.
  • composite  Inverse alpha-compositing per-frame (default for moving
               backgrounds). Instant; recovers the true semi-transparent
               background frame-by-frame.

Pipeline:
  1. ffmpeg  -> extract frames + audio
  2. temporal mean -> removal-residual AUTO-DETECTION of the spark (pos + size)
  3. per-clip α calibration (opacity zero-crossing)
  4. per-frame removal (lama or composite)
  5. ffmpeg  -> re-encode (CRF 15) + remux original audio

Verified on real Veo 1280x720 and 1920x1080 clips. The watermark size scales
with frame height (~0.0667·H => 48px @720p, 72px @1080p, 144px @4K) and Veo
stamps it in the bottom-right of a 16:9 frame; detection adapts to all of these.

Usage:
  python3 remove_watermark.py INPUT.mp4 [-o OUTPUT.mp4]
                              [--engine auto|lama|composite]
                              [--mask path/to/gemini_watermark.png]
                              [--crf 15] [--keep-temp]
"""
import argparse, os, sys, glob, subprocess, shutil, tempfile
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, binary_dilation

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MASK = os.path.normpath(os.path.join(HERE, "..", "assets", "gemini_watermark.png"))


# ----------------------------- helpers -----------------------------
def sh(cmd):
    return subprocess.run(cmd, capture_output=True, text=True)

def ffprobe(src, args):
    return sh(["ffprobe", "-v", "error"] + args + [src]).stdout.strip()

def load_mask(path):
    m = np.asarray(Image.open(path).convert("L"), np.float32)
    return m, float(m.max())

def mnorm(mask, mm, sz):
    return np.asarray(Image.fromarray(mask.astype(np.uint8)).resize((sz, sz), Image.BILINEAR), np.float32) / mm


# ----------------------- removal-residual detector -----------------------
# Finds where inverse-compositing the spark FLATTENS the image. Works on static
# AND moving backgrounds (operates on the temporal mean); resolution-adaptive.
def detect(mean_rgb, mask, mm):
    H, W = mean_rgb.shape[:2]
    lum = mean_rgb.mean(2)
    exp = 0.0667 * H                       # expected spark size for 16:9

    def score(cx, cy, sz, k=0.30):
        h = sz // 2; x0, y0 = cx - h, cy - h
        if x0 < 0 or y0 < 0 or x0 + sz > W or y0 + sz > H:
            return -1.0
        a = k * mnorm(mask, mm, sz)
        reg = lum[y0:y0 + sz, x0:x0 + sz]
        rem = np.clip((reg - a * 255) / (1 - a), 0, 255)
        r = max(2, sz // 8)
        e0 = np.abs(reg - gaussian_filter(reg, r))
        e1 = np.abs(rem - gaussian_filter(rem, r))
        w = mnorm(mask, mm, sz) > 0.05
        return float(e0[w].mean() - e1[w].mean())

    rx0, ry0 = int(W * 0.58), int(H * 0.56)      # Veo stamps far bottom-right (16:9)
    best = (-1.0, int(W * 0.88), int(H * 0.85), int(round(exp)))
    for sz in sorted({int(round(exp * 0.8)), int(round(exp)), int(round(exp * 1.2))}):
        cstep = max(3, sz // 12)
        for cy in range(ry0 + sz // 2, H - sz // 2, cstep):
            for cx in range(rx0 + sz // 2, W - sz // 2, cstep):
                s = score(cx, cy, sz)
                if s > best[0]:
                    best = (s, cx, cy, sz)
    _, bcx, bcy, bsz = best
    win = max(3, bsz // 12)
    for sz in range(int(round(exp * 0.72)), int(round(exp * 1.3)) + 1, 2):
        for dy in range(-win, win + 1):
            for dx in range(-win, win + 1):
                s = score(bcx + dx, bcy + dy, sz)
                if s > best[0]:
                    best = (s, bcx + dx, bcy + dy, sz)
    return best[1], best[2], best[3], best[0]    # cx, cy, sz, score


def calibrate_alpha(mean_rgb, mask, mm, cx, cy, sz):
    a = mnorm(mask, mm, sz); h = sz // 2
    reg = mean_rgb[cy - h:cy - h + sz, cx - h:cx - h + sz].mean(2)
    prev = None; kz = 0.30; k = 0.12
    while k <= 0.55:
        rem = np.clip((reg - k * a * 255) / (1 - k * a), 0, 255)
        hp = rem - gaussian_filter(rem, max(2, sz // 8))
        t = a - a.mean()
        ncc = float((hp * t).sum() / (np.sqrt((hp * hp).sum()) + 1e-6))
        if prev and prev[1] > 0 and ncc <= 0:
            kz = prev[0] + (0 - prev[1]) * (k - prev[0]) / (ncc - prev[1]); break
        prev = (k, ncc); k += 0.01
    return float(min(max(kz, 0.10), 0.55))


# --------------------------- removal engines ---------------------------
def composite_remove(frame, cx, cy, sz, alpha, k):
    """Inverse alpha-compositing: recover the true semi-transparent background."""
    out = frame.astype(np.float32)
    a = k * alpha; h = sz // 2; x0, y0 = cx - h, cy - h
    for c in range(3):
        out[y0:y0 + sz, x0:x0 + sz, c] = np.clip((out[y0:y0 + sz, x0:x0 + sz, c] - a * 255) / (1 - a), 0, 255)
    return out.astype(np.uint8)


class Lama:
    def __init__(self):
        import torch
        from simple_lama_inpainting.utils.util import prepare_img_and_mask  # noqa
        self.torch = torch
        self.prepare = prepare_img_and_mask
        mp = os.path.expanduser("~/.cache/torch/hub/checkpoints/big-lama.pt")
        if not os.path.exists(mp):
            try:
                from simple_lama_inpainting import SimpleLama
                SimpleLama()           # downloads the model file (CUDA-load may fail; ignored)
            except Exception:
                pass
        if not os.path.exists(mp):
            raise RuntimeError("Could not obtain big-lama.pt. Check internet / `pip install simple-lama-inpainting`.")
        self.model = torch.jit.load(mp, map_location="cpu").eval()

    def inpaint(self, image_np, mask_np):
        img, msk = self.prepare(Image.fromarray(image_np), Image.fromarray(mask_np), self.torch.device("cpu"))
        with self.torch.inference_mode():
            r = self.model(img, msk)[0].permute(1, 2, 0).detach().cpu().numpy()
        return np.clip(r * 255, 0, 255).astype(np.uint8)


# ------------------------------- main -------------------------------
def main():
    ap = argparse.ArgumentParser(description="Remove the Gemini/Veo watermark from a 16:9 video.")
    ap.add_argument("input")
    ap.add_argument("-o", "--output", default=None)
    ap.add_argument("--engine", choices=["auto", "lama", "composite"], default="auto")
    ap.add_argument("--mask", default=DEFAULT_MASK)
    ap.add_argument("--crf", default="15")
    ap.add_argument("--pos", default=None, help="override detection: 'cx,cy,size'")
    ap.add_argument("--keep-temp", action="store_true")
    args = ap.parse_args()

    src = os.path.abspath(args.input)
    if not os.path.exists(src):
        sys.exit("input not found: " + src)
    out = args.output or os.path.splitext(src)[0] + "_clean.mp4"
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        sys.exit("ffmpeg/ffprobe required (brew install ffmpeg).")
    mask, mm = load_mask(args.mask)

    W = int(ffprobe(src, ["-select_streams", "v:0", "-show_entries", "stream=width", "-of", "csv=p=0"]))
    H = int(ffprobe(src, ["-select_streams", "v:0", "-show_entries", "stream=height", "-of", "csv=p=0"]))
    fps = ffprobe(src, ["-select_streams", "v:0", "-show_entries", "stream=r_frame_rate", "-of", "csv=p=0"]) or "24"
    has_audio = ffprobe(src, ["-select_streams", "a", "-show_entries", "stream=index", "-of", "csv=p=0"]) != ""
    ar = W / H
    print(f"[info] {W}x{H}  {fps}fps  audio={has_audio}  aspect={ar:.3f}")
    if abs(ar - 16/9) > 0.05:
        print(f"[warn] this skill is tuned for 16:9; {ar:.2f}:1 may detect less reliably (use --pos to override).")

    tmp = tempfile.mkdtemp(prefix="gwmr_")
    fdir = os.path.join(tmp, "in"); odir = os.path.join(tmp, "out")
    os.makedirs(fdir); os.makedirs(odir)
    try:
        print("[1/5] extracting frames + audio…")
        sh(["ffmpeg", "-y", "-loglevel", "error", "-i", src, os.path.join(fdir, "%05d.png")])
        if has_audio:
            sh(["ffmpeg", "-y", "-loglevel", "error", "-i", src, "-vn", "-acodec", "copy", os.path.join(tmp, "audio.m4a")])
        frames = sorted(glob.glob(os.path.join(fdir, "*.png")))
        n = len(frames)
        if n == 0:
            sys.exit("no frames extracted.")
        print(f"      {n} frames")

        # temporal mean from a sample (for detection)
        samp = frames[:: max(1, n // 60)]
        mean_rgb = np.mean([np.asarray(Image.open(f).convert("RGB"), np.float32) for f in samp], 0)

        if args.pos:
            cx, cy, sz = [int(v) for v in args.pos.split(",")]; sc = float("nan")
        else:
            print("[2/5] detecting watermark…")
            cx, cy, sz, sc = detect(mean_rgb, mask, mm)
        k = calibrate_alpha(mean_rgb, mask, mm, cx, cy, sz)
        print(f"      spark @ ({cx},{cy})  size {sz}px  α≈{k:.3f}  score={sc:.2f}")

        # temporal variance in the region -> static vs moving
        h = sz // 2
        reg_stack = np.stack([np.asarray(Image.open(f).convert("RGB"), np.float32)[cy-h:cy+h, cx-h:cx+h] for f in samp])
        tvar = float(reg_stack.std(0).mean())
        engine = args.engine
        if engine == "auto":
            engine = "composite" if tvar > 28 else "lama"
        print(f"[3/5] region temporal-std {tvar:.1f} -> engine: {engine}")

        alpha = mnorm(mask, mm, sz)
        if engine == "lama":
            print("[4/5] LaMa per-frame inpaint (CPU; ~2-3s/frame)…")
            lama = Lama()
            cs = min(W, H, max(512, sz * 7))                  # context crop, fits frame
            X0 = min(max(cx - cs // 2, 0), W - cs); Y0 = min(max(cy - cs // 2, 0), H - cs)
            A = np.zeros((cs, cs), np.float32)
            ox, oy = cx - sz // 2 - X0, cy - sz // 2 - Y0
            A[oy:oy + sz, ox:ox + sz] = alpha
            holemask = binary_dilation(A > 0.04, iterations=max(6, round(sz * 0.16)))
            hm_img = (holemask * 255).astype(np.uint8)
            wpaste = np.clip(gaussian_filter(holemask.astype(np.float32), 2.0), 0, 1)[:, :, None]
            for i, f in enumerate(frames):
                fr = np.asarray(Image.open(f).convert("RGB")).astype(np.float32)
                crop = fr[Y0:Y0 + cs, X0:X0 + cs].astype(np.uint8)
                inp = lama.inpaint(crop, hm_img).astype(np.float32)
                if inp.shape[:2] != (cs, cs):
                    inp = np.asarray(Image.fromarray(inp.astype(np.uint8)).resize((cs, cs)), np.float32)
                fr[Y0:Y0 + cs, X0:X0 + cs] = fr[Y0:Y0 + cs, X0:X0 + cs] * (1 - wpaste) + inp * wpaste
                Image.fromarray(np.clip(fr, 0, 255).astype(np.uint8)).save(os.path.join(odir, "%05d.png" % (i + 1)))
                if i % 16 == 0:
                    print(f"      frame {i}/{n}")
        else:
            print("[4/5] inverse-composite per-frame…")
            for i, f in enumerate(frames):
                fr = np.asarray(Image.open(f).convert("RGB"))
                Image.fromarray(composite_remove(fr, cx, cy, sz, alpha, k)).save(os.path.join(odir, "%05d.png" % (i + 1)))

        print("[5/5] encoding + remux audio…")
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-framerate", fps, "-i", os.path.join(odir, "%05d.png")]
        if has_audio:
            cmd += ["-i", os.path.join(tmp, "audio.m4a"), "-c:a", "copy", "-map", "0:v", "-map", "1:a"]
        cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", str(args.crf), out]
        r = sh(cmd)
        if not os.path.exists(out):
            sys.exit("encode failed:\n" + r.stderr)
        print(f"[done] {out}  ({os.path.getsize(out)/1048576:.1f} MB)")
    finally:
        if not args.keep_temp:
            shutil.rmtree(tmp, ignore_errors=True)
        else:
            print("[temp]", tmp)


if __name__ == "__main__":
    main()
