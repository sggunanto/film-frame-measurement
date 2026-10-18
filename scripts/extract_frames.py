#!/usr/bin/env python3
"""
extract_frames.py — frame-accurate still extraction for the
"From Ruins to Threshold" corpus (Japanese animation, 1984-2022).

Reads the frozen logs (editions.csv, frame_log.csv), verifies that each video
file is the edition the logs describe, and extracts the logged frames as
lossless PNGs with a SHA-256 manifest.

Design notes
------------
* One decode pass per film. All of a film's target frames are selected in a
  single ffmpeg invocation via an OR'd `select` expression, so 18 frame-accurate
  stills cost one pass over the file instead of 18.
* Frame-accurate by construction: no input seeking (`-ss`) is used anywhere.
  Input seeking lands on keyframes and silently returns the wrong frame.
* Fails loudly. If the file's frame count, frame rate or resolution disagrees
  with editions.csv, nothing is extracted for that film.
* Writes nothing outside --out, and never modifies the source video.

The video files are never uploaded anywhere. This script runs locally on
legally held copies; only the extracted frames, or only the derived
measurements, need to leave the machine.

Usage
-----
    python extract_frames.py --editions editions.csv --frames frame_log.csv \
        --video-dir ./video --out ./frames [--film MON] [--dry-run]

    python extract_frames.py ... --probe-only      # report file properties, extract nothing
    python extract_frames.py ... --verify          # re-extract one frame per film, compare to manifest
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

VIDEO_EXTS = (".mkv", ".mp4", ".m4v", ".mov", ".avi", ".ts", ".m2ts", ".vob", ".webm")
FRAME_RATE_TOL = 0.001          # fps
RUNTIME_TOL_DEFAULT = 0         # frames


# ----------------------------------------------------------------- utilities
def die(msg: str, code: int = 2) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(code)


def run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True)


def need_binaries() -> tuple[str, str]:
    ff, fp = shutil.which("ffmpeg"), shutil.which("ffprobe")
    if not ff or not fp:
        die("ffmpeg and ffprobe must be on PATH. Install ffmpeg, then re-run.")
    return ff, fp


def ffmpeg_version(ffmpeg: str) -> str:
    out = run([ffmpeg, "-version"]).stdout.splitlines()
    return out[0].strip() if out else "unknown"


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def parse_rational(text: str) -> float | None:
    if not text or text in ("0/0", "N/A"):
        return None
    if "/" in text:
        num, den = text.split("/", 1)
        try:
            den_f = float(den)
            return float(num) / den_f if den_f else None
        except ValueError:
            return None
    try:
        return float(text)
    except ValueError:
        return None


# --------------------------------------------------------------------- probe
def probe(ffprobe: str, path: Path, count_frames: bool) -> dict:
    """Return the video stream's properties. count_frames=True is exact but slow."""
    args = [ffprobe, "-v", "error", "-select_streams", "v:0", "-of", "json"]
    if count_frames:
        args += ["-count_frames", "-show_entries",
                 "stream=nb_read_frames,r_frame_rate,avg_frame_rate,width,height,"
                 "pix_fmt,color_primaries,color_transfer,nb_frames"]
    else:
        args += ["-show_entries",
                 "stream=nb_frames,r_frame_rate,avg_frame_rate,width,height,"
                 "pix_fmt,color_primaries,color_transfer"]
    args += [str(path)]
    cp = run(args)
    if cp.returncode != 0:
        die(f"ffprobe failed on {path.name}: {cp.stderr.strip()[:300]}")
    streams = json.loads(cp.stdout).get("streams") or []
    if not streams:
        die(f"no video stream found in {path.name}")
    s = streams[0]
    nb = s.get("nb_read_frames") or s.get("nb_frames")
    try:
        nb = int(nb)
    except (TypeError, ValueError):
        nb = None
    return {
        "n_frames": nb,
        "frame_rate": parse_rational(s.get("r_frame_rate", "")),
        "avg_frame_rate": parse_rational(s.get("avg_frame_rate", "")),
        "width": s.get("width"),
        "height": s.get("height"),
        "pix_fmt": s.get("pix_fmt"),
        "color_primaries": s.get("color_primaries"),
        "color_transfer": s.get("color_transfer"),
        "frames_counted": bool(count_frames),
    }


def find_video(video_dir: Path, film_id: str, disc_or_file_id: str) -> Path | None:
    """Locate a film's file: exact name from the log first, then <film_id>.* ."""
    if disc_or_file_id:
        cand = video_dir / disc_or_file_id
        if cand.is_file():
            return cand
    hits = [p for p in sorted(video_dir.iterdir())
            if p.is_file() and p.suffix.lower() in VIDEO_EXTS
            and p.stem.upper().startswith(film_id.upper())]
    return hits[0] if len(hits) == 1 else None


# ----------------------------------------------------------------- log input
def read_csv(path: Path) -> list[dict]:
    if not path.is_file():
        die(f"missing log file: {path}")
    with open(path, newline="", encoding="utf-8-sig") as fh:
        rows = [r for r in csv.DictReader(fh)]
    if not rows:
        die(f"{path.name} has a header but no rows. Selection must be logged before extraction.")
    return rows


def validate_frame_rows(rows: list[dict]) -> list[dict]:
    required = {"frame_id", "seq_id", "film_id", "frame_role", "frame_index"}
    missing_cols = required - set(rows[0])
    if missing_cols:
        die(f"frame_log is missing columns: {sorted(missing_cols)}")
    clean, seen = [], set()
    for i, r in enumerate(rows, start=2):
        fid = (r.get("frame_id") or "").strip()
        if not fid:
            die(f"frame_log line {i}: empty frame_id")
        if fid in seen:
            die(f"frame_log line {i}: duplicate frame_id {fid!r}")
        seen.add(fid)
        try:
            r["frame_index"] = int(str(r["frame_index"]).strip())
        except ValueError:
            die(f"frame_log line {i}: frame_index {r['frame_index']!r} is not an integer")
        if r["frame_index"] < 0:
            die(f"frame_log line {i}: negative frame_index")
        if (r.get("frame_role") or "").strip() not in {"onset", "peak", "release"}:
            die(f"frame_log line {i}: frame_role must be onset, peak or release")
        clean.append(r)
    return clean


# ------------------------------------------------------------------ extract
def extract_film(ffmpeg: str, video: Path, targets: list[dict], out_dir: Path,
                 crop: tuple[int, int] | None, dry_run: bool) -> dict:
    """Extract one film's frames in a single decode pass. Returns a report dict."""
    targets = sorted(targets, key=lambda r: r["frame_index"])
    sel = "+".join(rf"eq(n\,{t['frame_index']})" for t in targets)
    vf = f"select='{sel}'"
    if crop:
        top, bottom = crop
        if top or bottom:
            vf = f"crop=iw:ih-{top + bottom}:0:{top}," + vf

    report = {"video": video.name, "n_requested": len(targets), "written": [], "missing": []}
    if dry_run:
        report["command"] = f"ffmpeg -i {video.name} -vf \"{vf}\" -fps_mode passthrough ..."
        return report

    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="extract_") as tmp:
        pattern = str(Path(tmp) / "tmp_%05d.png")
        cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin",
               "-i", str(video), "-vf", vf, "-fps_mode", "passthrough",
               "-compression_level", "0", "-pred", "mixed", pattern]
        cp = run(cmd)
        if cp.returncode != 0:
            die(f"ffmpeg failed on {video.name}: {cp.stderr.strip()[:400]}")
        produced = sorted(Path(tmp).glob("tmp_*.png"))
        if len(produced) != len(targets):
            report["missing"] = [t["frame_id"] for t in targets[len(produced):]]
            print(f"  WARNING: asked for {len(targets)} frames, ffmpeg returned "
                  f"{len(produced)}. The highest frame_index values are likely past "
                  f"the end of the file. Nothing is guessed; see the report.")
        # ffmpeg emits selected frames in ascending decode order, which equals
        # ascending frame_index. Pair them positionally.
        for src, tgt in zip(produced, targets):
            name = (f"{tgt['film_id']}_{tgt['seq_id']}_{tgt['frame_index']:07d}"
                    f"_{tgt['frame_role']}_{tgt['frame_id']}.png")
            dst = out_dir / name
            shutil.move(str(src), dst)
            report["written"].append({
                "frame_id": tgt["frame_id"], "seq_id": tgt["seq_id"],
                "film_id": tgt["film_id"], "frame_role": tgt["frame_role"],
                "frame_index": tgt["frame_index"],
                "timecode": tgt.get("timecode", ""),
                "path": str(dst.relative_to(out_dir.parent)),
                "bytes": dst.stat().st_size, "sha256": sha256(dst),
            })
    return report


# ---------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--editions", type=Path, default=Path("editions.csv"))
    ap.add_argument("--frames", type=Path, default=Path("frame_log.csv"))
    ap.add_argument("--video-dir", type=Path, default=Path("video"))
    ap.add_argument("--out", type=Path, default=Path("frames"))
    ap.add_argument("--film", action="append", default=None,
                    help="restrict to one film_id; repeatable")
    ap.add_argument("--dry-run", action="store_true",
                    help="verify logs and files, print what would be extracted")
    ap.add_argument("--probe-only", action="store_true",
                    help="report file properties against editions.csv and stop")
    ap.add_argument("--verify", action="store_true",
                    help="re-extract the first frame of each film and compare to the manifest")
    ap.add_argument("--count-frames", action="store_true",
                    help="exact frame count via full decode (slow, but the only reliable count)")
    ap.add_argument("--runtime-tol", type=int, default=RUNTIME_TOL_DEFAULT,
                    help="allowed |difference| in total frames before refusing (default 0)")
    ap.add_argument("--apply-crop", action="store_true",
                    help="apply crop_top/crop_bottom from editions.csv at extraction time "
                         "(default: keep full frame; frame_descriptors.py crops instead)")
    args = ap.parse_args()

    ffmpeg, ffprobe = need_binaries()
    editions = {r["film_id"].strip(): r for r in read_csv(args.editions)}
    frames = validate_frame_rows(read_csv(args.frames))

    films = sorted({r["film_id"].strip() for r in frames})
    if args.film:
        keep = {f.strip().upper() for f in args.film}
        films = [f for f in films if f.upper() in keep]
        if not films:
            die(f"--film {sorted(keep)} matches no film_id in {args.frames.name}")

    unknown = [f for f in films if f not in editions]
    if unknown:
        die(f"film_id(s) {unknown} appear in frame_log but not in editions.csv")

    print(f"{ffmpeg_version(ffmpeg)}")
    print(f"films: {', '.join(films)} | frames requested: "
          f"{sum(1 for r in frames if r['film_id'].strip() in films)}\n")

    all_reports, manifest_rows, blocked = {}, [], []
    for film in films:
        ed = editions[film]
        targets = [r for r in frames if r["film_id"].strip() == film]
        video = find_video(args.video_dir, film, (ed.get("disc_or_file_id") or "").strip())
        print(f"[{film}] {len(targets)} frame(s) requested")
        if video is None:
            print(f"  SKIP: no unique video file for {film} in {args.video_dir}. "
                  f"Name it {film}.mkv or set disc_or_file_id in editions.csv.")
            blocked.append(film)
            continue

        info = probe(ffprobe, video, count_frames=args.count_frames)
        print(f"  file: {video.name}")
        print(f"  probed: {info['n_frames']} frames @ {info['frame_rate']} fps, "
              f"{info['width']}x{info['height']}, {info['pix_fmt']}"
              f"{'' if info['frames_counted'] else '  (container metadata, not counted)'}")

        problems = []
        try:
            exp_frames = int(float(ed.get("runtime_frames") or 0))
        except ValueError:
            exp_frames = 0
        if exp_frames and info["n_frames"] is not None:
            delta = abs(info["n_frames"] - exp_frames)
            if delta > args.runtime_tol:
                problems.append(f"frame count {info['n_frames']} vs logged {exp_frames} "
                                f"(delta {delta} > tol {args.runtime_tol})")
        try:
            exp_fps = float(ed.get("frame_rate") or 0)
        except ValueError:
            exp_fps = 0.0
        if exp_fps and info["frame_rate"] and abs(info["frame_rate"] - exp_fps) > FRAME_RATE_TOL:
            problems.append(f"frame rate {info['frame_rate']:.5f} vs logged {exp_fps}")
        exp_res = (ed.get("resolution") or "").strip().lower()
        if exp_res and info["width"]:
            if exp_res.replace("×", "x") != f"{info['width']}x{info['height']}":
                problems.append(f"resolution {info['width']}x{info['height']} vs logged {exp_res}")

        over = [t["frame_id"] for t in targets
                if info["n_frames"] is not None and t["frame_index"] >= info["n_frames"]]
        if over:
            problems.append(f"frame_index past end of file for: {', '.join(over)}")

        if problems:
            print("  REFUSING to extract — this is not the logged edition:")
            for p in problems:
                print(f"    - {p}")
            print("  Fix editions.csv (or use the right file). Measurements are only "
                  "reproducible against a pinned edition.")
            blocked.append(film)
            continue
        if args.probe_only:
            print("  OK (probe-only)")
            continue

        crop = None
        if args.apply_crop:
            try:
                crop = (int(float(ed.get("crop_top") or 0)), int(float(ed.get("crop_bottom") or 0)))
            except ValueError:
                crop = None
        rep = extract_film(ffmpeg, video, targets, args.out, crop, args.dry_run)
        all_reports[film] = rep
        manifest_rows.extend(rep["written"])
        if args.dry_run:
            print(f"  DRY RUN: would run {rep['command']}")
        else:
            print(f"  wrote {len(rep['written'])}/{rep['n_requested']} PNG(s) to {args.out}")

    if args.probe_only:
        sys.exit(1 if blocked else 0)

    if manifest_rows and not args.dry_run:
        args.out.mkdir(parents=True, exist_ok=True)
        man = args.out / "manifest.csv"
        existing = []
        if man.is_file():
            with open(man, newline="", encoding="utf-8") as fh:
                existing = [r for r in csv.DictReader(fh)]
        by_id = {r["frame_id"]: r for r in existing}
        stamp = _dt.datetime.now().astimezone().isoformat(timespec="seconds")
        ver = ffmpeg_version(ffmpeg)
        for r in manifest_rows:
            r = dict(r, extracted_at=stamp, ffmpeg_version=ver)
            by_id[r["frame_id"]] = r
        cols = ["frame_id", "seq_id", "film_id", "frame_role", "frame_index", "timecode",
                "path", "bytes", "sha256", "extracted_at", "ffmpeg_version"]
        with open(man, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            for fid in sorted(by_id):
                w.writerow({c: by_id[fid].get(c, "") for c in cols})
        print(f"\nmanifest: {man} ({len(by_id)} frames total)")

    # ------------------------------------------------------ verify mode
    if args.verify and not args.dry_run:
        print("\nverify: re-extracting one frame per film")
        bad = 0
        for film, rep in all_reports.items():
            if not rep["written"]:
                continue
            first = rep["written"][0]
            video = find_video(args.video_dir, film,
                              (editions[film].get("disc_or_file_id") or "").strip())
            with tempfile.TemporaryDirectory() as tmp:
                tgt = [{"frame_id": first["frame_id"], "seq_id": first["seq_id"],
                        "film_id": film, "frame_role": first["frame_role"],
                        "frame_index": first["frame_index"], "timecode": ""}]
                rep2 = extract_film(ffmpeg, video, tgt, Path(tmp) / "f", None, False)
                same = rep2["written"] and rep2["written"][0]["sha256"] == first["sha256"]
                print(f"  [{film}] frame {first['frame_index']}: "
                      f"{'identical' if same else 'MISMATCH'}")
                bad += 0 if same else 1
        if bad:
            die(f"{bad} film(s) failed the re-extraction check — extraction is not deterministic "
                f"on this source; do not trust the descriptors.", 3)

    # --------------------------------------------------- completeness report
    print("\ncompleteness")
    total_req = total_got = 0
    for film in films:
        req = sum(1 for r in frames if r["film_id"].strip() == film)
        got = len(all_reports.get(film, {}).get("written", []))
        total_req += req
        total_got += got
        flag = "" if got == req else "   <-- incomplete"
        print(f"  {film}: {got}/{req}{flag}")
    print(f"  TOTAL: {total_got}/{total_req}")
    if blocked:
        print(f"\nblocked films (nothing extracted): {', '.join(blocked)}")
    if total_got != total_req and not args.dry_run:
        sys.exit(1)


if __name__ == "__main__":
    main()
