import os
import cv2
import time
import tempfile
import logging
import numpy as np
from typing import Dict, Any, List, Optional
from ultralytics import YOLO

from database import add_event, get_setting
from discord_notifier import send_discord_notification
from websocket_manager import ws_manager

logger = logging.getLogger("cctv_ai.image_video_processor")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

SECURITY_CLASS_IDS = [0, 1, 2, 3, 5, 7, 24, 26, 28]

CLASS_NAMES = {
    0: "Person",
    1: "Bicycle",
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck",
    24: "Backpack",
    26: "Handbag",
    28: "Suitcase"
}

CLASS_COLORS = {
    0: (255, 240, 0),      # Cyan (BGR)
    2: (129, 185, 16),     # Emerald (BGR)
    3: (129, 185, 16),     # Emerald (BGR)
    5: (129, 185, 16),     # Emerald (BGR)
    7: (129, 185, 16),     # Emerald (BGR)
    1: (11, 158, 245),     # Amber (BGR)
    24: (241, 102, 99),    # Indigo (BGR)
    26: (241, 102, 99),    # Indigo (BGR)
    28: (241, 102, 99),    # Indigo (BGR)
}


class ImageVideoProcessor:
    """
    Processor for static images and sampled video file analysis.
    Performs multi-class object detection (Persons, Vehicles, Bags),
    full-frame multi-object annotations, and real-time WebSocket progress reporting.
    """
    def __init__(self, screenshot_dir: str = "screenshots"):
        self.screenshot_dir = screenshot_dir
        os.makedirs(self.screenshot_dir, exist_ok=True)

    def process_image(
        self,
        image_bytes: bytes,
        model: YOLO,
        confidence: float = 0.15,
        camera_name: str = "Uploaded Image",
        webhook_url: Optional[str] = None,
        user_id: Optional[int] = 1
    ) -> Dict[str, Any]:
        """
        Processes a single uploaded image with complete multi-class detection & annotation.
        """
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            return {"success": False, "error": "Could not decode uploaded image file."}

        # Downscale ultra high-res images to max 640px for high-sensitivity multi-object detection
        h, w = img.shape[:2]
        max_dim = 640
        if max(h, w) > max_dim:
            scale = max_dim / float(max(h, w))
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        # Broadcast start progress over WebSocket
        try:
            ws_manager.broadcast_sync({
                "type": "PROCESSING_PROGRESS",
                "task": "IMAGE",
                "progress_pct": 10,
                "status_text": "Running YOLO multi-class object detection..."
            })
        except Exception:
            pass

        # Run YOLO detection with high sensitivity for all security classes
        results = model(img, conf=confidence, classes=SECURITY_CLASS_IDS, imgsz=640, verbose=False)


        person_count = 0
        vehicle_count = 0
        object_count = 0
        max_confidence = 0.0
        detections = []

        annotated_img = img.copy()

        if results and len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for box in boxes:
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())

                object_count += 1
                if cls_id == 0:
                    person_count += 1
                elif cls_id in [2, 3, 5, 7]:
                    vehicle_count += 1

                if conf > max_confidence:
                    max_confidence = conf

                cls_name = CLASS_NAMES.get(cls_id, "Object")
                detections.append({
                    "box": xyxy.tolist(),
                    "class": cls_name,
                    "class_id": cls_id,
                    "confidence": round(conf, 2)
                })

                # Draw bounding box & filled label badge for ALL detected objects
                x1, y1, x2, y2 = xyxy
                color = CLASS_COLORS.get(cls_id, (0, 240, 255))

                cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 2)
                label = f"{cls_name} {int(conf * 100)}%"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                cv2.rectangle(annotated_img, (x1, max(0, y1 - 22)), (x1 + tw + 8, max(0, y1)), color, -1)
                cv2.putText(annotated_img, label, (x1 + 4, max(14, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

        # Save complete annotated image
        timestamp = int(time.time())
        filename = f"img_detection_{timestamp}_{person_count}p_{vehicle_count}v.jpg"
        filepath = os.path.join(self.screenshot_dir, filename)
        cv2.imwrite(filepath, annotated_img)

        # Log event in DB if at least 1 object detected
        event_id = None
        if object_count > 0:
            event_id = add_event(
                camera_name=camera_name,
                track_id=1,
                object_class="person" if person_count > 0 else "object",
                confidence=max_confidence,
                screenshot_filename=filename,
                screenshot_path=filepath,
                source_type="IMAGE",
                user_id=user_id
            )

            # Non-blocking Discord notification in background thread
            target_webhook = webhook_url or get_setting("DISCORD_WEBHOOK_URL", os.getenv("DISCORD_WEBHOOK_URL", ""))
            if target_webhook:
                import threading
                def _bg_discord_send():
                    try:
                        send_discord_notification(
                            image_path=filepath,
                            track_id=1,
                            camera_name=camera_name,
                            object_class=f"Detected {person_count} Persons, {vehicle_count} Vehicles",
                            confidence=max_confidence,
                            webhook_url=target_webhook
                        )
                    except Exception as err:
                        logger.warning(f"Background Discord notification error: {err}")
                threading.Thread(target=_bg_discord_send, daemon=True).start()


            # Broadcast completion & event over WebSocket
            try:
                ws_manager.broadcast_sync({
                    "type": "NEW_EVENT",
                    "event": {
                        "id": event_id,
                        "datetime_str": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "camera_name": camera_name,
                        "track_id": 1,
                        "object_class": f"{person_count} Persons, {vehicle_count} Vehicles",
                        "confidence": round(max_confidence, 2),
                        "screenshot_filename": filename,
                        "screenshot_path": filepath,
                        "source_type": "IMAGE"
                    }
                })
            except Exception as ws_err:
                logger.warning(f"WebSocket broadcast error: {ws_err}")

        # Broadcast completion progress
        try:
            ws_manager.broadcast_sync({
                "type": "PROCESSING_PROGRESS",
                "task": "IMAGE",
                "progress_pct": 100,
                "status_text": f"Complete! Found {person_count} Persons, {vehicle_count} Vehicles.",
                "latest_screenshot_url": f"/screenshots/{filename}"
            })
        except Exception:
            pass

        return {
            "success": True,
            "source_type": "IMAGE",
            "person_count": person_count,
            "vehicle_count": vehicle_count,
            "total_objects": object_count,
            "max_confidence": round(max_confidence, 2) if max_confidence > 0 else 0.0,
            "filename": filename,
            "annotated_image_url": f"/screenshots/{filename}",
            "event_id": event_id,
            "detections": detections
        }

    def process_video(
        self,
        video_bytes: bytes,
        model: YOLO,
        sample_interval_sec: float = 1.0,
        confidence: float = 0.15,
        camera_name: str = "Uploaded Video",
        webhook_url: Optional[str] = None,
        user_id: Optional[int] = 1
    ) -> Dict[str, Any]:
        """
        Processes video file with frame sampling & ByteTrack tracking:
        - Samples 1 frame/sec
        - Annotates ALL detected persons, vehicles, and objects in every frame
        - Broadcasts real-time frame-by-frame progress & keyframes over WebSocket
        """
        temp_dir = tempfile.gettempdir()
        temp_video_path = os.path.join(temp_dir, f"upload_video_{int(time.time())}.mp4")
        with open(temp_video_path, "wb") as f:
            f.write(video_bytes)

        cap = cv2.VideoCapture(temp_video_path)
        if not cap.isOpened():
            if os.path.exists(temp_video_path):
                os.remove(temp_video_path)
            return {"success": False, "error": "Could not open or decode uploaded video file."}

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        duration_sec = total_frames / fps if fps > 0 else 0.0

        frame_step = max(1, int(fps * sample_interval_sec))
        total_sampled_frames = max(1, (total_frames + frame_step - 1) // frame_step)

        unique_track_ids = set()
        alerted_track_ids = set()
        events_generated = []
        max_overall_confidence = 0.0
        sampled_frame_count = 0
        current_frame_idx = 0

        # Reset model tracker predictor state
        if hasattr(model, "predictor") and model.predictor is not None:
            if hasattr(model.predictor, "trackers") and model.predictor.trackers:
                for trk in model.predictor.trackers:
                    if hasattr(trk, "reset"):
                        trk.reset()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            current_frame_idx += 1
            if (current_frame_idx - 1) % frame_step != 0:
                continue

            # Downscale frame to max 640px for high sensitivity tracking on CPU
            h, w = frame.shape[:2]
            max_dim = 640
            if max(h, w) > max_dim:
                scale = max_dim / float(max(h, w))
                frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

            sampled_frame_count += 1
            pct = int((sampled_frame_count / total_sampled_frames) * 100)

            # Run multi-class tracking on sampled frame using ByteTrack
            try:
                results = model.track(
                    frame,
                    conf=confidence,
                    classes=SECURITY_CLASS_IDS,
                    imgsz=640,
                    persist=True,
                    tracker="bytetrack.yaml",
                    verbose=False
                )
            except Exception:
                results = model(frame, conf=confidence, classes=SECURITY_CLASS_IDS, imgsz=640, verbose=False)


            frame_persons = 0
            frame_vehicles = 0

            # Process detection results if boxes are present
            if results and len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                for box in boxes:
                    conf = float(box.conf[0].cpu().numpy())
                    cls_id = int(box.cls[0].cpu().numpy())
                    cls_name = CLASS_NAMES.get(cls_id, "Object")
                    track_id = int(box.id[0].cpu().numpy()) if box.id is not None else 1
                    xyxy = box.xyxy[0].cpu().numpy().astype(int).tolist()

                    if conf > max_overall_confidence:
                        max_overall_confidence = conf

                    if cls_id == 0:
                        frame_persons += 1
                        unique_track_ids.add(track_id)
                        if track_id not in alerted_track_ids:
                            alerted_track_ids.add(track_id)
                            new_person_tracks.append({
                                "track_id": track_id,
                                "box": xyxy,
                                "conf": conf,
                                "cls_name": cls_name
                            })
                    elif cls_id in [2, 3, 5, 7]:
                        frame_vehicles += 1

                    frame_boxes_data.append({
                        "track_id": track_id,
                        "cls_id": cls_id,
                        "cls_name": cls_name,
                        "conf": conf,
                        "box": xyxy
                    })

            # 1. MAIN FRAME: Draw ALL detected objects onto annotated_frame for main player display
            annotated_frame = frame.copy()
            for item in frame_boxes_data:
                x1, y1, x2, y2 = item["box"]
                cls_id = item["cls_id"]
                color = CLASS_COLORS.get(cls_id, (0, 240, 255))
                lbl = f"ID:{item['track_id']} {item['cls_name']} {int(item['conf'] * 100)}%"
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                cv2.rectangle(annotated_frame, (x1, max(0, y1 - 22)), (x1 + tw + 8, max(0, y1)), color, -1)
                cv2.putText(annotated_frame, lbl, (x1 + 4, max(14, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

            timestamp = int(time.time())
            main_filename = f"vid_main_{sampled_frame_count}_{timestamp}.jpg"
            main_filepath = os.path.join(self.screenshot_dir, main_filename)
            cv2.imwrite(main_filepath, annotated_frame)

            # 2. INDIVIDUAL EVENT SCREENSHOTS: Save dedicated screenshot highlighting ONLY THAT SPECIFIC PERSON
            for ntr in new_person_tracks:
                tid = ntr["track_id"]
                x1, y1, x2, y2 = ntr["box"]
                conf = ntr["conf"]

                single_track_frame = frame.copy()
                color = (255, 240, 0)  # Bright Cyan
                cv2.rectangle(single_track_frame, (x1, y1), (x2, y2), color, 2)
                label = f"Person #{tid} {int(conf * 100)}%"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                cv2.rectangle(single_track_frame, (x1, max(0, y1 - 22)), (x1 + tw + 8, max(0, y1)), color, -1)
                cv2.putText(single_track_frame, label, (x1 + 4, max(14, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

                event_filename = f"person_track_{tid}_{timestamp}.jpg"
                event_filepath = os.path.join(self.screenshot_dir, event_filename)
                cv2.imwrite(event_filepath, single_track_frame)

                event_id = add_event(
                    camera_name=camera_name,
                    track_id=tid,
                    object_class="person",
                    confidence=conf,
                    screenshot_filename=event_filename,
                    screenshot_path=event_filepath,
                    source_type="VIDEO",
                    user_id=user_id
                )

                event_data = {
                    "id": event_id,
                    "track_id": tid,
                    "confidence": round(conf, 2),
                    "timestamp_sec": round(current_frame_idx / fps, 1),
                    "filename": event_filename,
                    "screenshot_url": f"/screenshots/{event_filename}"
                }
                events_generated.append(event_data)

                # Broadcast new event over WebSocket
                try:
                    ws_manager.broadcast_sync({
                        "type": "NEW_EVENT",
                        "event": {
                            "id": event_id,
                            "datetime_str": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "camera_name": camera_name,
                            "track_id": tid,
                            "object_class": f"Person ID #{tid}",
                            "confidence": round(conf, 2),
                            "screenshot_filename": event_filename,
                            "screenshot_path": event_filepath,
                            "source_type": "VIDEO"
                        }
                    })
                except Exception:
                    pass

            # Broadcast real-time progress update over WebSocket for EVERY sampled frame
            try:
                ws_manager.broadcast_sync({
                    "type": "PROCESSING_PROGRESS",
                    "task": "VIDEO",
                    "progress_pct": min(99, pct),
                    "sampled_frame": sampled_frame_count,
                    "total_sampled": total_sampled_frames,
                    "persons_detected": len(unique_track_ids),
                    "vehicles_detected": frame_vehicles,
                    "status_text": f"Processing Frame {sampled_frame_count}/{total_sampled_frames} ({pct}%) • {len(unique_track_ids)} Persons Tracked",
                    "latest_screenshot_url": f"/screenshots/{main_filename}"
                })
            except Exception:
                pass


        cap.release()
        if os.path.exists(temp_video_path):
            try:
                os.remove(temp_video_path)
            except Exception:
                pass

        # Final progress 100% broadcast
        try:
            ws_manager.broadcast_sync({
                "type": "PROCESSING_PROGRESS",
                "task": "VIDEO",
                "progress_pct": 100,
                "status_text": f"Video analysis complete! Tracked {len(unique_track_ids)} unique persons."
            })
        except Exception:
            pass

        return {
            "success": True,
            "source_type": "VIDEO",
            "duration_sec": round(duration_sec, 1),
            "sampled_frames_processed": sampled_frame_count,
            "unique_persons_tracked": len(unique_track_ids),
            "total_events_generated": len(events_generated),
            "max_confidence": round(max_overall_confidence, 2) if max_overall_confidence > 0 else 0.0,
            "events": events_generated
        }


image_video_processor = ImageVideoProcessor()
