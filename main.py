import argparse
from pathlib import Path
from sports_tracker.detector import Detector
from sports_tracker.tracker import SortTracker
from sports_tracker.visualizer import Visualizer
from sports_tracker.utils.video_utils import read_frames, save_video

MODEL_PATH = "yolov8n.pt"


def main() -> None:
    parser = argparse.ArgumentParser(description="Sports Tracker CLI")
    parser.add_argument("input", type=Path, help="Input video file")
    parser.add_argument("--output", type=Path, default=Path("output/result.mp4"), help="Output video path")
    parser.add_argument("--confidence", type=float, default=0.35, help="Detection confidence threshold")
    args = parser.parse_args()

    detector = Detector(MODEL_PATH, conf_threshold=args.confidence)
    tracker = SortTracker()
    visualizer = Visualizer()

    frames = []
    for frame in read_frames(args.input):
        detections = detector.detect(frame)
        tracks = tracker.update(detections)
        frames.append(visualizer.draw(frame, tracks))

    save_video(frames, args.output)
    print(f"Saved result to {args.output}")


if __name__ == "__main__":
    main()
