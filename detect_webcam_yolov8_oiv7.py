# save as detect_webcam_yolov8_oiv7.py
"""
Real-time YOLOv8 (Open Images V7) webcam detector.
Defaults to yolov8x-oiv7.pt (largest OIV7 model).
Usage examples:
  python detect_webcam_yolov8_oiv7.py
  python detect_webcam_yolov8_oiv7.py --model yolov8m-oiv7.pt --conf 0.45
  python detect_webcam_yolov8_oiv7.py --filter "glasses,pen,wallet"
"""

import argparse
import time
import sys

try:
    from ultralytics import YOLO
except Exception as e:
    print("ERROR: ultralytics not found. Install with: pip install ultralytics")
    raise e

import cv2
import numpy as np
import torch

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--camera", type=int, default=0, help="Camera index (default 0)")
    p.add_argument("--model", type=str, default="yolov8x-oiv7.pt", help="YOLOv8 OIV7 model name or path")
    p.add_argument("--conf", type=float, default=0.5, help="Confidence threshold (0..1)")
    p.add_argument("--imgsz", type=int, default=640, help="YOLO inference image size (px)")
    p.add_argument("--width", type=int, default=640, help="Camera width (px)")
    p.add_argument("--height", type=int, default=480, help="Camera height (px)")
    p.add_argument("--filter", type=str, default=None,
                   help="Comma-separated object names to detect (e.g. \"glasses,pen,wallet\"). Optional.")
    p.add_argument("--save", type=str, default=None, help="Optional: path to save annotated output video (mp4)")
    return p.parse_args()

def map_filter_to_ids(model, filter_str):
    if not filter_str:
        return None
    requested = [s.strip().lower() for s in filter_str.split(",") if s.strip()]
    # model.names is a dict: {id: "class_name"}
    name_to_id = {v.lower(): k for k, v in model.names.items()}
    class_ids = []
    missing = []
    for req in requested:
        if req in name_to_id:
            class_ids.append(name_to_id[req])
        else:
            # try substring match (e.g., user typed "glasses" but model name might be "eyeglasses")
            matches = [nm for nm in name_to_id if req in nm]
            if len(matches) == 1:
                class_ids.append(name_to_id[matches[0]])
            elif len(matches) > 1:
                # pick the best match (first) but inform user
                class_ids.append(name_to_id[matches[0]])
                print(f"Note: '{req}' matched multiple names {matches}. Using '{matches[0]}'.")
            else:
                missing.append(req)
    if missing:
        print(f"Warning: these requested names were NOT found in model class list: {missing}")
    if not class_ids:
        print("No valid class names found from filter — running without class filtering.")
        return None
    print(f"Filtering to class IDs: {class_ids} -> {[model.names[i] for i in class_ids]}")
    return class_ids

def main():
    args = parse_args()

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}  |  Model: {args.model}")

    # Load model (will download if not present)
    model = YOLO(args.model)
    if device == "cuda":
        try:
            model.to("cuda")
        except Exception as e:
            print("Warning: couldn't move model to cuda. Continuing on CPU.")
            print(e)

    # Open camera
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"ERROR: Cannot open camera {args.camera}")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    # Optional save setup
    writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.save, fourcc, 20.0, (args.width, args.height))
        print(f"Saving output to: {args.save}")

    # Map filter names to class ids (if given)
    class_ids = map_filter_to_ids(model, args.filter)

    print("Press ESC or 'q' to quit.")
    prev_time = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Frame read failed, exiting")
            break

        # Run detection; pass classes if we have them
        try:
            if class_ids is not None:
                results = model(frame, conf=args.conf, imgsz=args.imgsz, classes=class_ids)
            else:
                results = model(frame, conf=args.conf, imgsz=args.imgsz)
        except Exception as e:
            print("Inference error:", e)
            break

        # Annotated frame
        annotated = results[0].plot()  # returns numpy array in BGR or RGB? ultralytics returns BGR for OpenCV compat
        # In some versions plot returns RGB; ensure it's BGR for imshow by converting if needed:
        if annotated.ndim == 3 and annotated.shape[2] == 3:
            # Heuristic: if values look like floats 0-1 or RGB, convert to BGR uint8
            if annotated.dtype != np.uint8:
                annotated = (annotated * 255).astype(np.uint8)
        # Put FPS
        cur_time = time.time()
        fps = 1.0 / (cur_time - prev_time + 1e-6)
        prev_time = cur_time
        cv2.putText(annotated, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow("YOLOv8 OIV7 Live (press q or ESC to quit)", annotated)

        if writer is not None:
            # ensure writer frame size matches annotated frame size
            h, w = annotated.shape[:2]
            # if writer was created with different size, it may fail — we attempt to write the resized frame
            if (w, h) != (args.width, args.height):
                frame_to_write = cv2.resize(annotated, (args.width, args.height))
            else:
                frame_to_write = annotated
            writer.write(frame_to_write)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord("q"):
            break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
