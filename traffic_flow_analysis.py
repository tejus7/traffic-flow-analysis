import argparse
import os
import json
import cv2
import numpy as np
from ultralytics import YOLO
from sort import Sort
import csv

# ---------------- Lane Annotation ----------------
def annotate_lanes(video_path, lanes_json="lanes.json"):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Could not read video for annotation")

    lanes = []
    current_polygon = []
    lane_index = 0

    instructions = [
        "Draw Lane 1 (Leftmost): Click points, Enter to finish.",
        "Draw Lane 2 (Middle): Click points, Enter to finish.",
        "Draw Lane 3 (Rightmost): Click points, Enter to finish.",
    ]

    def mouse_callback(event, x, y, flags, param):
        nonlocal current_polygon
        if event == cv2.EVENT_LBUTTONDOWN:
            current_polygon.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN:
            if current_polygon:
                current_polygon.pop()

    cv2.namedWindow("Annotate Lanes")
    cv2.setMouseCallback("Annotate Lanes", mouse_callback)

    while True:
        display_frame = frame.copy()

        # Draw completed lanes in green
        for idx, lane in enumerate(lanes):
            cv2.polylines(display_frame, [np.array(lane, dtype=np.int32)], isClosed=True, color=(0, 255, 0), thickness=2)
            cv2.putText(display_frame, f"Lane {idx+1}", lane[0], cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Draw current polygon in red
        if current_polygon:
            cv2.polylines(display_frame, [np.array(current_polygon, dtype=np.int32)], isClosed=False, color=(0, 0, 255), thickness=2)

        # Show instructions
        cv2.putText(display_frame, instructions[lane_index], (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow("Annotate Lanes", display_frame)
        key = cv2.waitKey(1) & 0xFF

        if key == 13:  # Enter
            if len(current_polygon) >= 3:
                lanes.append(current_polygon.copy())
                current_polygon.clear()
                lane_index += 1
                if lane_index == 3:
                    break
            else:
                print("⚠ Need at least 3 points before pressing Enter.")

        elif key in (ord('r'), ord('R')):
            current_polygon.clear()

        elif key == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()

    if len(lanes) != 3:
        raise RuntimeError(f"Please draw exactly 3 polygons (you drew {len(lanes)}).")

    with open(lanes_json, "w") as f:
        json.dump(lanes, f)

    print(f"✅ Saved lanes to {lanes_json}")
    return lanes


# ---------------- Load or Annotate ----------------
def load_or_annotate_lanes(video_path, annotate=False, lanes_json="lanes.json"):
    print(f"[DEBUG] annotate flag received = {annotate}")
    if annotate:
        print("[INFO] Annotation mode: please draw 3 lanes...")
        return annotate_lanes(video_path, lanes_json)

    if os.path.exists(lanes_json):
        print(f"[INFO] Loading lanes from {lanes_json}")
        with open(lanes_json) as f:
            return json.load(f)
    else:
        print("[INFO] No lanes.json found, starting annotation...")
        return annotate_lanes(video_path, lanes_json)


# ---------------- Vehicle Detection & Counting ----------------
def point_in_polygon(point, polygon):
    return cv2.pointPolygonTest(np.array(polygon, dtype=np.int32), point, False) >= 0

def process_video(video_path, lanes):
    print("[INFO] Loading YOLO model...")
    model = YOLO("yolov8n.pt")
    tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(3))
    height = int(cap.get(4))

    out = cv2.VideoWriter("output_annotated.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    vehicle_counts = [0, 0, 0]
    seen_ids = set()

    with open("output_events.csv", mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["VehicleID", "Lane", "Frame", "Timestamp"])

        frame_num = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_num += 1
            timestamp = frame_num / fps

            results = model(frame, verbose=False)
            detections = []

            for r in results:
                for box in r.boxes:
                    cls = int(box.cls[0])
                    if model.names[cls] in ["car", "truck", "bus", "motorbike"]:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = float(box.conf[0])
                        detections.append([x1, y1, x2, y2, conf])

            tracks = tracker.update(np.array(detections))

            for track in tracks:
                x1, y1, x2, y2, track_id = map(int, track)
                center = ((x1 + x2) // 2, (y1 + y2) // 2)

                lane_number = None
                for i, lane in enumerate(lanes):
                    if point_in_polygon(center, lane):
                        lane_number = i + 1
                        break

                if lane_number and (track_id, lane_number) not in seen_ids:
                    vehicle_counts[lane_number - 1] += 1
                    seen_ids.add((track_id, lane_number))
                    writer.writerow([track_id, lane_number, frame_num, round(timestamp, 2)])

                color = (0, 255, 0) if lane_number else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"ID {track_id} L{lane_number if lane_number else '-'}", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            for lane in lanes:
                cv2.polylines(frame, [np.array(lane, dtype=np.int32)], True, (255, 0, 0), 2)

            for i, count in enumerate(vehicle_counts, start=1):
                cv2.putText(frame, f"Lane {i}: {count}", (20, 50 * i), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            out.write(frame)

            if frame_num % int(fps) == 0:  # Show progress every second
                print(f"[INFO] Processed {frame_num} frames...")

    cap.release()
    out.release()

    print("\n[SUMMARY] Vehicle Counts:")
    for i, count in enumerate(vehicle_counts, start=1):
        print(f"Lane {i}: {count}")


# ---------------- Main ----------------
# ---------------- Main ----------------
def main():
    parser = argparse.ArgumentParser(description="Traffic Flow Analysis")
    parser.add_argument("--annotate", action="store_true", help="Draw lanes manually")
    parser.add_argument("--video", type=str, default="traffic_source.mp4", help="Path to video file")
    args = parser.parse_args()

    if not os.path.exists(args.video):
        raise FileNotFoundError(f"Video file '{args.video}' not found.")

    # Always get lanes (either by drawing or loading)
    lanes = load_or_annotate_lanes(args.video, annotate=args.annotate, lanes_json="lanes.json")
    
    print(f"[INFO] Lanes loaded: {len(lanes)} polygons. Starting vehicle detection...")
    
    # Always process the video after lanes are available
    process_video(args.video, lanes)


if __name__ == "__main__":
    main()
