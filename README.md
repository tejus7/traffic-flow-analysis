# Traffic Flow Analysis (3-Lane Vehicle Counting)

A complete, from-scratch implementation to **detect, track, and count vehicles per lane** from a road video.  
Meets the brief: download the given YouTube video, use a COCO model (YOLO), track objects across frames to avoid double counting, export CSV, overlay counts on the video, and print a final summary.
A Python-based system that detects and tracks vehicles in 3 lanes, counts them in real time, and exports results to CSV and annotated video.
---

## 1) Features

- **Video downloader** (uses `yt-dlp`) from: `https://www.youtube.com/watch?v=MNn9qKG2UFI`
- **Vehicle detection** via **YOLOv8** (Ultralytics) using COCO classes for `car`, `motorcycle`, `bus`, `truck`.
- **Original lightweight tracker** (ID-stable, IOU + center-distance hybrid). Simple, robust, and easy to reason about.
- **Three-lane definition**: interactively annotate 3 polygons **once**, saved to `lanes.json`. Reuse them every run.
- **Per-lane counting**: each unique track is counted at first entry into that lane polygon (no duplicates).
- **CSV export** (`output_events.csv`): Vehicle ID, Lane, Frame, Timestamp (sec).
- **Annotated MP4** (`output_annotated.mp4`): overlays lanes, IDs, and live lane counts.
- **Real-time friendly**: resize frames, confidence threshold, model variant are adjustable CLI flags.
- **Reproducible**: single main script, no hidden magic.

---

## 2) Quick Start (Step-by-step)

> Prereqs: Python 3.9+ recommended; a GPU is optional but improves speed. On first run, YOLO will download weights.

### A) Set up environment
```bash
# 1) create & activate a virtualenv (recommended)
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 2) install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### B) First run: download video & annotate lanes
```bash
# This downloads the video, opens the first frame for annotation, and then runs counting end-to-end.
python traffic_flow_analysis.py --download --annotate
```
**How annotation works**
- A frame pops up. You will draw 3 lane polygons **in order**: Lane 1, Lane 2, Lane 3.
- **Click** points to define a polygon. Press **ENTER** to close each polygon.
- Press **Z** to undo last point. Press **R** to reset the current polygon.
- After you finish a lane, it auto-advances to the next. After Lane 3, it saves to `lanes.json`.

### C) Typical run (reuse existing lanes.json, skip re-download)
```bash
python traffic_flow_analysis.py
```
Optional flags:
```bash
# Use a different YOLO weights file, adjust confidence, speed-up by resizing, run display window:
python traffic_flow_analysis.py --model yolov8n.pt --conf 0.3 --max-width 1280 --show

# Run on an already-downloaded local video instead of YouTube:
python traffic_flow_analysis.py --video-path path/to/video.mp4
```

---

## 3) Outputs

- `output_events.csv` — vehicle-level entries: VehicleID, Lane, Frame, Timestamp (sec)
- `output_annotated.mp4` — annotated video with lanes, IDs, and live counts
- Console summary at the end: total per lane

---

## 4) Demo video (1–2 minutes)

Once `output_annotated.mp4` exists, trim a short clip (requires ffmpeg installed):
```bash
# Take a 90-second clip starting at 00:00:10
ffmpeg -ss 00:00:10 -i output_annotated.mp4 -t 00:01:30 -c copy demo_clip.mp4
```
Upload `demo_clip.mp4` to Google Drive (or similar) and set sharing to “Anyone with the link can view”.

---

## 5) Repository structure
```
traffic-flow-analysis/
├─ traffic_flow_analysis.py
├─ requirements.txt
├─ README.md
└─ lanes.json              # auto-created after first annotation
```

---

## 6) Command reference

```
python traffic_flow_analysis.py [--download] [--video-path PATH]
                               [--annotate] [--model YOLO_WEIGHTS]
                               [--conf FLOAT] [--max-width INT] [--show]
```

- `--download`: fetches the YouTube video locally if not present
- `--video-path`: path to a local video (overrides default)
- `--annotate`: run lane annotation UI before processing
- `--model`: YOLO weights name or path (e.g., `yolov8n.pt`, `yolov8s.pt`)
- `--conf`: detection confidence (default 0.25)
- `--max-width`: downscale frames to this width for speed (keeps aspect)
- `--show`: display live window in addition to writing the MP4

---

## 7) Tips for accuracy & speed

- Prefer `yolov8s.pt` or `yolov8m.pt` on a good GPU for accuracy.
- If CPU-only, try `--max-width 960` or `--max-width 720`.
- Keep lane polygons tight around the actual road lanes’ perspective.
- You can re-run with `--annotate` anytime to redraw lanes.

---

## 8) Technical summary (for your submission notes)

- **Detector**: YOLOv8 (COCO) using classes: `car`, `motorcycle`, `bus`, `truck`.
- **Tracker**: a custom hybrid of IOU matching and centroid-distance gating to keep IDs stable without heavy dependencies.
- **Counting logic**: first time a track’s center enters a lane polygon, it is counted for that lane and recorded to CSV.
- **Outputs**: CSV + annotated MP4 + printed summary.
- **Performance**: frame resizing and confidence threshold are configurable. GPU is auto-used if available.

---

## 9) Common issues

- **No window appears** during annotation: some environments (e.g., SSH without display) block GUI. Run locally or on a machine with display.
- **Slow inference**: reduce `--max-width`, use a smaller YOLO model, or enable GPU.
- **Wrong lane counts**: re-run with `--annotate` to fine-tune polygons.

---

## 10) License

MIT — do whatever you want; attribution appreciated.
