
import argparse
from pathlib import Path
import cv2
import numpy as np


def iou(box_a, box_b):
    if box_a is None or box_b is None:
        return 0.0
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    x1 = max(ax, bx)
    y1 = max(ay, by)
    x2 = min(ax + aw, bx + bw)
    y2 = min(ay + ah, by + bh)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter <= 0:
        return 0.0
    union = aw * ah + bw * bh - inter
    return inter / float(union + 1e-6)


def expand_box(box, image_shape, pad_x=0.35, pad_y=0.45):
    h, w = image_shape[:2]
    x, y, bw, bh = box
    cx = x + bw / 2.0
    cy = y + bh / 2.0
    nw = bw * (1.0 + 2.0 * pad_x)
    nh = bh * (1.0 + 2.0 * pad_y)
    nx = int(max(0, cx - nw / 2.0))
    ny = int(max(0, cy - nh / 2.0))
    nx2 = int(min(w, cx + nw / 2.0))
    ny2 = int(min(h, cy + nh / 2.0))
    return nx, ny, max(1, nx2 - nx), max(1, ny2 - ny)


def pad_crop_box(box, image_shape, left=0.05, right=0.11, top=0.10, bottom=0.12):
    """Asymmetric padding: slightly more room on the right/bottom so edge digits do not get cut."""
    h, w = image_shape[:2]
    x, y, bw, bh = box
    x1 = max(0, int(x - bw * left))
    y1 = max(0, int(y - bh * top))
    x2 = min(w, int(x + bw + bw * right))
    y2 = min(h, int(y + bh + bh * bottom))
    return x1, y1, max(1, x2 - x1), max(1, y2 - y1)


def nms_candidates(candidates, iou_threshold=0.35):
    if not candidates:
        return []
    candidates = sorted(candidates, key=lambda item: item[0], reverse=True)
    kept = []
    for cand in candidates:
        score, box, debug = cand
        if all(iou(box, kept_box) < iou_threshold for _, kept_box, _ in kept):
            kept.append(cand)
    return kept


def preprocess_masks(gray):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    norm = clahe.apply(gray)
    blur = cv2.bilateralFilter(norm, 7, 55, 55)

    # White-ish plate region candidates.
    white = cv2.inRange(blur, 145, 255)
    white = cv2.morphologyEx(
        white,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (17, 5)),
        iterations=2,
    )

    # Character-like dark strokes on bright background.
    blackhat = cv2.morphologyEx(
        blur,
        cv2.MORPH_BLACKHAT,
        cv2.getStructuringElement(cv2.MORPH_RECT, (19, 7)),
    )
    bh = cv2.normalize(blackhat, None, 0, 255, cv2.NORM_MINMAX)
    _, chars = cv2.threshold(bh, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    chars = cv2.morphologyEx(
        chars,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3)),
        iterations=1,
    )

    combined = cv2.bitwise_and(
        white,
        cv2.dilate(chars, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3)), iterations=1)
    )
    combined = cv2.morphologyEx(
        combined,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (21, 5)),
        iterations=2,
    )
    return white, chars, combined


def count_char_components(chars_patch, min_area=20):
    cnts, _ = cv2.findContours(chars_patch, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    n = 0
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        aspect = w / float(h + 1e-6)
        if area >= min_area and 0.10 <= aspect <= 1.2 and h >= 0.28 * chars_patch.shape[0]:
            n += 1
    return n


def refine_crop_with_chars(gray_full, box):
    if box is None:
        return None
    x, y, w, h = box
    pad = max(4, int(min(w, h) * 0.08))
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(gray_full.shape[1], x + w + pad)
    y1 = min(gray_full.shape[0], y + h + pad)
    roi = gray_full[y0:y1, x0:x1]
    _, chars, _ = preprocess_masks(roi)
    cnts, _ = cv2.findContours(chars, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pts = []
    for c in cnts:
        cx, cy, cw, ch = cv2.boundingRect(c)
        area = cw * ch
        aspect = cw / float(ch + 1e-6)
        if area >= 20 and 0.10 <= aspect <= 1.2:
            pts.append((cx, cy, cw, ch))
    if len(pts) < 3:
        return box
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x2s = [p[0] + p[2] for p in pts]
    y2s = [p[1] + p[3] for p in pts]
    rx0 = max(0, min(xs) - 8)
    ry0 = max(0, min(ys) - 10)
    rx1 = min(roi.shape[1], max(x2s) + 12)  # extra room on right
    ry1 = min(roi.shape[0], max(y2s) + 10)
    refined = (x0 + rx0, y0 + ry0, max(1, rx1 - rx0), max(1, ry1 - ry0))
    return refined


def candidate_score(gray_roi, white_mask, chars_mask, x, y, w, h, prev_box_local=None, mode='video'):
    area = w * h
    if h <= 0 or w <= 0:
        return None
    aspect = w / float(h)
    if mode == 'image':
        if not (2.2 <= aspect <= 6.4):
            return None
        if not (1200 <= area <= 65000):
            return None
    else:
        if not (2.0 <= aspect <= 6.8):
            return None
        if not (1200 <= area <= 100000):
            return None

    patch = gray_roi[y:y+h, x:x+w]
    if patch.size == 0:
        return None

    std_intensity = float(patch.std())
    edges = cv2.Canny(patch, 70, 180)
    edge_density = float((edges > 0).mean())

    dark_chars = (patch < 130).astype(np.uint8)
    char_ratio = float(dark_chars.mean())
    transitions = np.abs(np.diff((patch < 150).astype(np.int16), axis=1))
    transition_density = float((transitions > 0).mean())

    mask_patch = white_mask[y:y+h, x:x+w]
    chars_patch = chars_mask[y:y+h, x:x+w]
    white_ratio = float((mask_patch > 0).mean()) if mask_patch.size else 0.0
    chars_ratio = float((chars_patch > 0).mean()) if chars_patch.size else 0.0
    char_components = count_char_components(chars_patch, min_area=max(16, int(area * 0.0025)))

    center_x = x + w / 2.0
    center_y = y + h / 2.0
    center_penalty = abs(center_x - gray_roi.shape[1] / 2.0) / gray_roi.shape[1]
    lower_bonus = center_y / gray_roi.shape[0]

    score = (
        110.0 * edge_density
        + 150.0 * transition_density
        + 40.0 * char_ratio
        + 45.0 * chars_ratio
        + 50.0 * white_ratio
        + 12.0 * min(char_components, 8)
        + 0.006 * area
        + 0.18 * std_intensity
        + 9.0 * lower_bonus
        - 26.0 * center_penalty
    )

    if char_ratio < 0.06 or edge_density < 0.024:
        score -= 40.0
    if white_ratio < 0.18:
        score -= 28.0
    if chars_ratio < 0.05:
        score -= 28.0
    if char_components < 3:
        score -= 35.0

    ideal_aspect = 4.3 if mode == 'image' else 4.1
    score -= 8.0 * abs(aspect - ideal_aspect)

    if prev_box_local is not None:
        score += 55.0 * iou((x, y, w, h), prev_box_local)

    return score


def detect_plate(image: np.ndarray, prev_box=None, mode='video'):
    H, W = image.shape[:2]
    gray_full = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    rois = []
    if mode == 'image':
        rois.append((int(W * 0.22), int(H * 0.56), int(W * 0.56), int(H * 0.28), 1.40))
        rois.append((int(W * 0.16), int(H * 0.50), int(W * 0.68), int(H * 0.34), 1.05))
    else:
        rois.append((int(W * 0.16), int(H * 0.42), int(W * 0.68), int(H * 0.50), 1.05))
        rois.append((int(W * 0.10), int(H * 0.40), int(W * 0.80), int(H * 0.55), 0.86))

    if prev_box is not None and mode == 'video':
        px, py, pw, ph = expand_box(prev_box, image.shape, pad_x=0.45, pad_y=0.65)
        rois.insert(0, (px, py, pw, ph, 1.50))

    candidates = []
    best_debug = None

    for rx, ry, rw, rh, roi_weight in rois:
        rx = max(0, min(rx, W - 1))
        ry = max(0, min(ry, H - 1))
        rw = max(1, min(rw, W - rx))
        rh = max(1, min(rh, H - ry))

        roi = image[ry:ry+rh, rx:rx+rw]
        gray = gray_full[ry:ry+rh, rx:rx+rw]
        white_mask, chars_mask, combined = preprocess_masks(gray)
        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        prev_local = None
        if prev_box is not None:
            px, py, pw, ph = prev_box
            prev_local = (px - rx, py - ry, pw, ph)

        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            score = candidate_score(gray, white_mask, chars_mask, x, y, w, h, prev_local, mode=mode)
            if score is None:
                continue
            score *= roi_weight
            candidates.append((score, (x + rx, y + ry, w, h), {'roi': roi, 'white_mask': white_mask, 'chars_mask': chars_mask, 'combined': combined}))

        if best_debug is None:
            best_debug = {'roi': roi, 'white_mask': white_mask, 'chars_mask': chars_mask, 'combined': combined}

    best_box = None
    debug = best_debug if best_debug is not None else {'roi': image, 'white_mask': gray_full, 'chars_mask': gray_full, 'combined': gray_full}
    if candidates:
        candidates = nms_candidates(candidates, iou_threshold=0.30)
        candidates.sort(key=lambda item: item[0], reverse=True)
        best_score, best_box, debug = candidates[0]
        if best_score < (48 if mode == 'image' else 38):
            best_box = None
        elif mode == 'image':
            best_box = refine_crop_with_chars(gray_full, best_box)

    return best_box, debug


def smooth_box(current_box, stable_box, alpha=0.34):
    if current_box is None:
        return stable_box
    if stable_box is None:
        return current_box
    x = int((1 - alpha) * stable_box[0] + alpha * current_box[0])
    y = int((1 - alpha) * stable_box[1] + alpha * current_box[1])
    w = int((1 - alpha) * stable_box[2] + alpha * current_box[2])
    h = int((1 - alpha) * stable_box[3] + alpha * current_box[3])
    return (x, y, w, h)


def draw_result(image: np.ndarray, box, label='Detected Plate'):
    output = image.copy()
    if box is not None:
        x, y, w, h = box
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 4)
        cv2.putText(output, label, (x, max(35, y - 12)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
    return output


def process_image(input_path: str, output_dir: str):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(input_path)
    if image is None:
        raise FileNotFoundError(f'Could not read image: {input_path}')

    box, debug = detect_plate(image, mode='image')
    result = draw_result(image, box)

    stem = Path(input_path).stem
    result_path = output_dir / f'{stem}_detected.png'
    cv2.imwrite(str(result_path), result)

    crop_path = None
    if box is not None:
        x, y, w, h = pad_crop_box(box, image.shape, left=0.05, right=0.12, top=0.10, bottom=0.12)
        crop = image[y:y+h, x:x+w]
        crop_path = output_dir / f'{stem}_plate_crop.png'
        cv2.imwrite(str(crop_path), crop)

    cv2.imwrite(str(output_dir / f'{stem}_mask_white.png'), debug['white_mask'])
    cv2.imwrite(str(output_dir / f'{stem}_mask_chars.png'), debug['chars_mask'])
    cv2.imwrite(str(output_dir / f'{stem}_mask_combined.png'), debug['combined'])
    print(f'Saved annotated image to: {result_path}')
    if crop_path:
        print(f'Saved plate crop to: {crop_path}')
    else:
        print('No plate candidate found.')


def process_video(input_path: str, output_path: str, resize_width=None):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise FileNotFoundError(f'Could not open video: {input_path}')

    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if resize_width:
        scale = resize_width / float(width)
        width = int(resize_width)
        height = int(height * scale)
    else:
        scale = 1.0

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_idx = 0
    found = 0
    last_raw_box = None
    stable_box = None
    miss_streak = 0
    detect_streak = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if scale != 1.0:
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

        proposal, _ = detect_plate(frame, prev_box=stable_box if stable_box is not None else last_raw_box, mode='video')

        box = proposal
        if proposal is not None:
            if stable_box is not None and iou(proposal, stable_box) < 0.03 and detect_streak >= 2:
                # reject sudden jumps for a few frames
                miss_streak += 1
                if miss_streak <= 2:
                    box = stable_box
            else:
                stable_box = smooth_box(proposal, stable_box, alpha=0.36)
                last_raw_box = proposal
                box = stable_box
                miss_streak = 0
                detect_streak += 1
        else:
            detect_streak = 0
            miss_streak += 1
            if stable_box is not None and miss_streak <= 4:
                box = stable_box
            elif miss_streak > 4:
                stable_box = None
                last_raw_box = None
                box = None

        annotated = draw_result(frame, box, label='Plate')
        cv2.putText(annotated, f'Frame: {frame_idx}', (20, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        writer.write(annotated)
        if box is not None:
            found += 1
        frame_idx += 1

    cap.release()
    writer.release()
    print(f'Saved annotated video to: {output_path}')
    print(f'Frames with plate candidate: {found}/{frame_idx}')


def build_parser():
    parser = argparse.ArgumentParser(description='Refined classical license-plate localization demo.')
    sub = parser.add_subparsers(dest='mode', required=True)

    p_img = sub.add_parser('image', help='Run detection on an image')
    p_img.add_argument('--input', required=True, help='Input image path')
    p_img.add_argument('--output-dir', default='results/image_outputs', help='Directory for output images')

    p_vid = sub.add_parser('video', help='Run detection on a video')
    p_vid.add_argument('--input', required=True, help='Input video path')
    p_vid.add_argument('--output', default='results/video_outputs/demo_dashcam_plate_annotated.mp4', help='Output annotated video path')
    p_vid.add_argument('--resize-width', type=int, default=None, help='Optional resize width for more stable video processing')

    return parser


if __name__ == '__main__':
    args = build_parser().parse_args()
    if args.mode == 'image':
        process_image(args.input, args.output_dir)
    elif args.mode == 'video':
        process_video(args.input, args.output, resize_width=args.resize_width)
