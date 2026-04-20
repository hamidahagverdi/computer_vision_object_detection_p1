import argparse
from pathlib import Path
import sys
import cv2
import numpy as np
import requests
from ultralytics import YOLO

DEFAULT_MODEL_URL = "https://huggingface.co/yasirfaizahmed/license-plate-object-detection/resolve/main/best.pt"
DEFAULT_MODEL_PATH = Path("models/license_plate_yolov8_best.pt")


def ensure_model(model_path: Path, model_url: str) -> Path:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    if model_path.exists() and model_path.stat().st_size > 1024 * 1024:
        return model_path

    print(f"Downloading YOLO plate model to: {model_path}")
    try:
        with requests.get(model_url, stream=True, timeout=120) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            downloaded = 0
            with open(model_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if not chunk:
                        continue
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = 100.0 * downloaded / total
                        print(f"  {pct:5.1f}%", end="\r")
        print("Download complete.            ")
    except Exception as e:
        if model_path.exists():
            model_path.unlink(missing_ok=True)
        raise RuntimeError(
            "Failed to download the YOLO license-plate model. "
            "Check your internet connection, or manually place a .pt plate model at "
            f"{model_path}. Original error: {e}"
        )
    return model_path


def pick_best_plate(result, conf_threshold: float = 0.20):
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return None

    candidates = []
    for box in boxes:
        conf = float(box.conf[0].item()) if box.conf is not None else 0.0
        if conf < conf_threshold:
            continue
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
        w = max(1, x2 - x1)
        h = max(1, y2 - y1)
        aspect = w / float(h)
        area = w * h
        score = conf
        if 2.0 <= aspect <= 6.5:
            score += 0.15
        score += min(area / 150000.0, 0.15)
        candidates.append((score, (x1, y1, x2, y2), conf))

    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0]


def pad_box(x1, y1, x2, y2, shape, padx=0.05, pady=0.12):
    h, w = shape[:2]
    bw = x2 - x1
    bh = y2 - y1
    nx1 = max(0, int(x1 - bw * padx))
    ny1 = max(0, int(y1 - bh * pady))
    nx2 = min(w, int(x2 + bw * padx))
    ny2 = min(h, int(y2 + bh * pady))
    return nx1, ny1, nx2, ny2


def run_image(args):
    image_path = Path(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    model = YOLO(str(ensure_model(Path(args.model), args.model_url)))
    result = model.predict(source=str(image_path), conf=args.conf, verbose=False)[0]
    best = pick_best_plate(result, args.conf)

    annotated = image.copy()
    stem = image_path.stem
    if best is None:
        out_path = out_dir / f"{stem}_yolo_detected.png"
        cv2.imwrite(str(out_path), annotated)
        print(f"No YOLO plate found. Saved image anyway: {out_path}")
        return

    _, (x1, y1, x2, y2), conf = best
    px1, py1, px2, py2 = pad_box(x1, y1, x2, y2, image.shape)
    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(annotated, f"YOLO plate {conf:.2f}", (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    crop = image[py1:py2, px1:px2]
    out_detected = out_dir / f"{stem}_yolo_detected.png"
    out_crop = out_dir / f"{stem}_yolo_plate_crop.png"
    cv2.imwrite(str(out_detected), annotated)
    cv2.imwrite(str(out_crop), crop)
    print(f"Saved YOLO annotated image to: {out_detected}")
    print(f"Saved YOLO plate crop to: {out_crop}")


def run_video(args):
    video_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    target_w = args.resize_width or width
    scale = target_w / float(width)
    target_h = int(height * scale)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (target_w, target_h))

    model = YOLO(str(ensure_model(Path(args.model), args.model_url)))

    total = 0
    hits = 0
    prev = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        total += 1

        frame = cv2.resize(frame, (target_w, target_h)) if target_w != width else frame
        result = model.predict(source=frame, conf=args.conf, verbose=False)[0]
        best = pick_best_plate(result, args.conf)

        if best is not None:
            _, (x1, y1, x2, y2), conf = best
            if prev is not None:
                px1, py1, px2, py2 = prev
                alpha = 0.35
                x1 = int(alpha * x1 + (1 - alpha) * px1)
                y1 = int(alpha * y1 + (1 - alpha) * py1)
                x2 = int(alpha * x2 + (1 - alpha) * px2)
                y2 = int(alpha * y2 + (1 - alpha) * py2)
            prev = (x1, y1, x2, y2)
            hits += 1
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"YOLO plate {conf:.2f}", (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        writer.write(frame)

    cap.release()
    writer.release()
    print(f"Saved YOLO annotated video to: {out_path}")
    print(f"Frames with YOLO plate detection: {hits}/{total}")


def main():
    parser = argparse.ArgumentParser(description="YOLO-based license plate detection for image and video")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    img = subparsers.add_parser("image", help="Run YOLO plate detection on a single image")
    img.add_argument("--input", required=True)
    img.add_argument("--output-dir", default="results/yolo_image_outputs")
    img.add_argument("--conf", type=float, default=0.20)
    img.add_argument("--model", default=str(DEFAULT_MODEL_PATH))
    img.add_argument("--model-url", default=DEFAULT_MODEL_URL)

    vid = subparsers.add_parser("video", help="Run YOLO plate detection on a video")
    vid.add_argument("--input", required=True)
    vid.add_argument("--output", required=True)
    vid.add_argument("--resize-width", type=int, default=960)
    vid.add_argument("--conf", type=float, default=0.20)
    vid.add_argument("--model", default=str(DEFAULT_MODEL_PATH))
    vid.add_argument("--model-url", default=DEFAULT_MODEL_URL)

    args = parser.parse_args()
    try:
        if args.mode == "image":
            run_image(args)
        else:
            run_video(args)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
