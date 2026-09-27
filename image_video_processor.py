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


class ImageVideoProcessor:
    """
    Dedicated processor for static images and sampled video file analysis.
    Designed for fast execution on CPU cloud environments (e.g., Render)
    without needing continuous live streams.
    """
    def __init__(self, screenshot_dir: str = "screenshots"):
        self.screenshot_dir = screenshot_dir
        os.makedirs(self.screenshot_dir, exist_ok=True)

    def process_image(
        self,
        image_bytes: bytes,
        model: YOLO,
        confidence: float = 0.50,
        camera_name: str = "Uploaded Image",
        webhook_url: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Processes a single uploaded image:
        - Runs YOLO person detection
        - Annotates bounding boxes and confidence scores
        - Saves result screenshot
        - Saves record to database (source_type='IMAGE')
        - Triggers optional Discord notification
        """
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            return {"success": False, "error": "Could not decode uploaded image file."}

        # Run YOLO detection for Person (COCO class 0)
        results = model(img, conf=confidence, classes=[0], verbose=False)

        person_count = 0
        max_confidence = 0.0
        detections = []

        annotated_img = img.copy()

        if results and len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for box in boxes:
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0].cpu().numpy())

                person_count += 1
                if conf > max_confidence:
                    max_confidence = conf

                detections.append({
                    "box": xyxy.tolist(),
                    "confidence": round(conf, 2)
                })

                # Draw bounding box & label
                x1, y1, x2, y2 = xyxy
                cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 127), 2)

                label = f"Person {int(conf * 100)}%"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(annotated_img, (x1, max(0, y1 - 25)), (x1 + tw + 10, max(0, y1)), (0, 255, 127), -1)
                cv2.putText(annotated_img, label, (x1 + 5, max(15, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        # Save annotated image
        timestamp = int(time.time())
        filename = f"img_detection_{timestamp}_{person_count}p.jpg"
        filepath = os.path.join(self.screenshot_dir, filename)
        cv2.imwrite(filepath, annotated_img)

        # Log event in DB if at least 1 person detected
        event_id = None
        if person_count > 0:
            event_id = add_event(
                camera_name=camera_name,
                track_id=1,
                object_class="person",
                confidence=max_confidence,
                screenshot_filename=filename,
                screenshot_path=filepath,
                source_type="IMAGE"
            )

            # Check Discord notification settings
            target_webhook = webhook_url or get_setting("DISCORD_WEBHOOK_URL", os.getenv("DISCORD_WEBHOOK_URL", ""))
            if target_webhook:
                try:
                    send_discord_notification(
                        image_path=filepath,
                        track_id=1,
                        camera_name=camera_name,
                        object_class="person",
                        confidence=max_confidence,
                        webhook_url=target_webhook
                    )
                except Exception as err:
                    logger.warning(f"Failed to send Discord alert for image upload: {err}")

            # Push WebSocket event update
            try:
                event_payload = {
                    "type": "NEW_EVENT",
                    "event": {
                        "id": event_id,
                        "datetime_str": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "camera_name": camera_name,
                        "track_id": 1,
                        "object_class": "person",
                        "confidence": round(max_confidence, 2),
                        "screenshot_filename": filename,
                        "screenshot_path": filepath,
                        "source_type": "IMAGE"
                    }
                }
                ws_manager.broadcast_sync(event_payload)
            except Exception as ws_err:
                logger.warning(f"WebSocket broadcast exception for image: {ws_err}")

        return {
            "success": True,
            "source_type": "IMAGE",
            "person_count": person_count,
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
        confidence: float = 0.50,
        camera_name: str = "Uploaded Video",
        webhook_url: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Processes an uploaded video file with frame sampling & ByteTrack tracking:
        - Samples 1 frame/sec (or configured sample_interval_sec)
        - Runs YOLO + ByteTrack across sampled frames
        - Captures evidence screenshot per new track ID detected
        - Logs events to database (source_type='VIDEO')
        - Sends Discord notifications for distinct tracked persons
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

        unique_track_ids = set()
        alerted_track_ids = set()
        events_generated = []
        max_overall_confidence = 0.0
        sampled_frame_count = 0

        current_frame_idx = 0

        # Reset model tracker predictor state if available
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

            sampled_frame_count += 1

            # Run tracking on sampled frame using ByteTrack tracker
            try:
                results = model.track(
                    frame,
                    conf=confidence,
                    classes=[0],
                    persist=True,
                    tracker="bytetrack.yaml",
                    verbose=False
                )
            except Exception as e:
                # Fallback to standard detect if tracker error
                results = model(frame, conf=confidence, classes=[0], verbose=False)

            if results and len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                for box in boxes:
                    conf = float(box.conf[0].cpu().numpy())
                    if conf > max_overall_confidence:
                        max_overall_confidence = conf

                    # Check track id
                    track_id = int(box.id[0].cpu().numpy()) if box.id is not None else 1
                    unique_track_ids.add(track_id)

                    # Trigger event if new track ID seen
                    if track_id not in alerted_track_ids:
                        alerted_track_ids.add(track_id)

                        # Draw annotated bounding box & Track ID
                        annotated_frame = frame.copy()
                        xyxy = box.xyxy[0].cpu().numpy().astype(int)
                        x1, y1, x2, y2 = xyxy

                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (59, 130, 246), 2)
                        label = f"ID: {track_id} | {int(conf * 100)}%"
                        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                        cv2.rectangle(annotated_frame, (x1, max(0, y1 - 25)), (x1 + tw + 10, max(0, y1)), (59, 130, 246), -1)
                        cv2.putText(annotated_frame, label, (x1 + 5, max(15, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                        # Save keyframe screenshot
                        timestamp = int(time.time())
                        filename = f"vid_track_{track_id}_{timestamp}.jpg"
                        filepath = os.path.join(self.screenshot_dir, filename)
                        cv2.imwrite(filepath, annotated_frame)

                        # Add event to DB
                        event_id = add_event(
                            camera_name=camera_name,
                            track_id=track_id,
                            object_class="person",
                            confidence=conf,
                            screenshot_filename=filename,
                            screenshot_path=filepath,
                            source_type="VIDEO"
                        )

                        event_data = {
                            "id": event_id,
                            "track_id": track_id,
                            "confidence": round(conf, 2),
                            "timestamp_sec": round(current_frame_idx / fps, 1),
                            "filename": filename,
                            "screenshot_url": f"/screenshots/{filename}"
                        }
                        events_generated.append(event_data)

                        # Send Discord Notification
                        target_webhook = webhook_url or get_setting("DISCORD_WEBHOOK_URL", os.getenv("DISCORD_WEBHOOK_URL", ""))
                        if target_webhook:
                            try:
                                send_discord_notification(
                                    image_path=filepath,
                                    track_id=track_id,
                                    camera_name=camera_name,
                                    object_class="person",
                                    confidence=conf,
                                    webhook_url=target_webhook
                                )
                            except Exception as err:
                                logger.warning(f"Failed to send Discord alert for video track {track_id}: {err}")

                        # Broadcast WebSocket Event
                        try:
                            ws_manager.broadcast_sync({
                                "type": "NEW_EVENT",
                                "event": {
                                    "id": event_id,
                                    "datetime_str": time.strftime("%Y-%m-%d %H:%M:%S"),
                                    "camera_name": camera_name,
                                    "track_id": track_id,
                                    "object_class": "person",
                                    "confidence": round(conf, 2),
                                    "screenshot_filename": filename,
                                    "screenshot_path": filepath,
                                    "source_type": "VIDEO"
                                }
                            })
                        except Exception as ws_err:
                            logger.warning(f"WebSocket broadcast error for video frame: {ws_err}")

        cap.release()
        if os.path.exists(temp_video_path):
            try:
                os.remove(temp_video_path)
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
