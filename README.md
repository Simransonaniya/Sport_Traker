# Sports Tracker

A simple sports video tracking project using YOLOv8 for object detection and a SORT-style tracker for object IDs.

## Project Structure

- `sports_tracker/app.py` - Streamlit web UI entrypoint
- `sports_tracker/main.py` - CLI terminal entrypoint
- `sports_tracker/detector.py` - YOLOv8 detector wrapper
- `sports_tracker/tracker.py` - SORT-like tracking logic
- `sports_tracker/visualizer.py` - draws bounding boxes and IDs
- `sports_tracker/utils/video_utils.py` - video load/save helpers
- `sports_tracker/requirements.txt` - Python dependencies
- `sports_tracker/packages.txt` - Linux system dependencies
- `sports_tracker/.streamlit/config.toml` - Streamlit theme/config
- `video.mp4` - sample input placeholder
- `output/result.mp4` - processed output placeholder

## Setup

1. Create and activate a Python environment.
2. Install Python dependencies:

```bash
pip install -r sports_tracker/requirements.txt
```

3. Ensure `yolov8n.pt` is available in the project root, or update the `MODEL_PATH` constant in `sports_tracker/app.py` and `sports_tracker/main.py`.

## Run Streamlit UI

```bash
streamlit run sports_tracker/app.py
```

Open the browser UI and upload a video to process.

## Run CLI

```bash
python -m sports_tracker.main video.mp4 --output output/result.mp4
```

## Notes

- The tracker is a simple implementation intended for demonstration.
- Replace `video.mp4` with your own input video file.
- Output is saved to `output/result.mp4` by default.
