import argparse
import subprocess
from pathlib import Path
import cv2
import numpy as np


def label_image(img, text):
    canvas = img.copy()
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 42), (30, 30, 30), -1)
    cv2.putText(canvas, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return canvas


def load_or_blank(path: Path, shape=(480, 640, 3), text="Missing output"):
    img = cv2.imread(str(path))
    if img is not None:
        return img
    blank = np.full(shape, 40, dtype=np.uint8)
    cv2.putText(blank, text, (20, shape[0] // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    return blank


def main():
    parser = argparse.ArgumentParser(description="Run classical and YOLO methods on the same image and build a comparison image.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--classical-out", default="results/image_outputs")
    parser.add_argument("--yolo-out", default="results/yolo_image_outputs")
    parser.add_argument("--comparison", default="results/method_comparison.png")
    args = parser.parse_args()

    img_path = Path(args.input)
    stem = img_path.stem

    subprocess.run([
        "python3", "src/detect_plate.py", "image",
        "--input", str(img_path),
        "--output-dir", args.classical_out,
    ], check=False)

    subprocess.run([
        "python3", "src/detect_plate_yolo.py", "image",
        "--input", str(img_path),
        "--output-dir", args.yolo_out,
    ], check=False)

    original = load_or_blank(img_path)
    classical = load_or_blank(Path(args.classical_out) / f"{stem}_detected.png", shape=original.shape)
    yolo = load_or_blank(Path(args.yolo_out) / f"{stem}_yolo_detected.png", shape=original.shape)

    h = min(original.shape[0], classical.shape[0], yolo.shape[0])
    def resize_to_h(img):
        w = int(img.shape[1] * h / img.shape[0])
        return cv2.resize(img, (w, h))

    original = label_image(resize_to_h(original), "Original")
    classical = label_image(resize_to_h(classical), "Classical CV")
    yolo = label_image(resize_to_h(yolo), "YOLO")

    montage = np.hstack([original, classical, yolo])
    out_path = Path(args.comparison)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), montage)
    print(f"Saved comparison image to: {out_path}")


if __name__ == "__main__":
    main()
