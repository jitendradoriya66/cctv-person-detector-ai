import os
os.environ["OPENCV_LOG_LEVEL"] = "OFF"
import cv2
import time
import threading
from typing import Optional, Dict, Any, List
from ultralytics import YOLO
from database import add_event, get_setting, save_setting
from discord_notifier import send_discord_notification
from websocket_manager import ws_manager

class CameraManager:
    """
    Background Camera & YOLO Object Detection Engine.
    Handles OpenCV RTSP/Webcam streaming, object detection, tracking,
    MJPEG frame generation, database logging, and Discord notifications.
    """
    def __init__(self, model_path: str = "yolo26n.pt", screenshot_dir: str = "screenshots"):
        self.model_path = model_path
        self.screenshot_dir = screenshot_dir
        os.makedirs(self.screenshot_dir, exist_ok=True)
        
        self.model: Optional[YOLO] = None
        self.is_running = False
        self.camera_thread: Optional[threading.Thread] = None
        
        # Lock for thread-safe access to frame bytes
        self.frame_lock = threading.Lock()
        self.current_frame_bytes: Optional[bytes] = None
        
        # Configuration parameters
        self.camera_source: Any = 0
        self.camera_name: str = "Entrance Camera"
        self.confidence_threshold: float = 0.50
        self.target_class_ids: List[int] = [0]  # COCO class 0 = person
        
        # Analytics & Stats
        self.alerted_ids = set()
        self.current_fps: float = 0.0
        self.active_persons_in_frame: int = 0
        self.total_session_events: int = 0
        self.last_event_time: Optional[str] = None
        self.latest_event_info: Optional[Dict[str, Any]] = None
        
        # Load model lazily
        self._load_model()
        
    def _load_model(self):
        try:
            print(f"🔄 Loading YOLO model from {self.model_path}...")
            self.model = YOLO(self.model_path)
            print("✅ YOLO Model loaded successfully!")
        except Exception as e:
            print(f"❌ Failed to load YOLO model: {e}")

    def start(self, source: Any = None, confidence: float = None, camera_name: str = None) -> bool:
        # If camera is running and new settings are provided, stop current stream first
        if self.is_running:
            print("🔄 Dynamic camera source or confidence update requested. Restarting stream...")
            self.stop()

        if source is not None:
            if source == "demo1":
                self.camera_source = "https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/people-detection.mp4"
                self.camera_name = "Demo CCTV Stream 1"
            elif source == "demo2":
                self.camera_source = "https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/head-pose-face-detection-female.mp4"
                self.camera_name = "Demo CCTV Stream 2"
            elif isinstance(source, str) and source.isdigit():
                self.camera_source = int(source)
            else:
                self.camera_source = source

        if confidence is not None:
            self.confidence_threshold = confidence
            
        if camera_name is not None:
            self.camera_name = camera_name
            
        self.is_running = True
        self.camera_thread = threading.Thread(target=self._process_stream, daemon=True)
        self.camera_thread.start()
        print(f"🟢 Camera stream thread started on source: {self.camera_source}")
        return True

    def stop(self) -> bool:
        if not self.is_running:
            print("⚠️ Camera is not running.")
            return True
            
        self.is_running = False
        if self.camera_thread and self.camera_thread.is_alive():
            self.camera_thread.join(timeout=3.0)
        print("🔴 Camera stream stopped.")
        return True

    def _process_stream(self):
        source_target = self.camera_source

        # If on Linux cloud server (Render) without physical webcam /dev/video0, use demo stream directly
        if os.name != 'nt':
            if isinstance(source_target, int) or (isinstance(source_target, str) and str(source_target).isdigit()):
                dev_path = f"/dev/video{source_target}"
                if not os.path.exists(dev_path):
                    print(f"ℹ️ Cloud environment without '{dev_path}'. Streaming online CCTV demo video...")
                    source_target = "https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/people-detection.mp4"
                    self.camera_name = "Demo CCTV Stream"

        cap = cv2.VideoCapture(source_target)
        if not cap.isOpened():
            print(f"⚠️ Video source '{source_target}' unavailable. Attempting online CCTV demo stream fallback...")
            fallback_url = "https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/people-detection.mp4"
            cap = cv2.VideoCapture(fallback_url)
            if not cap.isOpened():
                print(f"❌ Failed to open fallback video stream: {fallback_url}")
                self.is_running = False
                return
            else:
                self.camera_name = "Demo CCTV Stream"

        frame_count = 0
        start_time = time.time()

        while self.is_running:
            ret, frame = cap.read()
            if not ret:
                # If reading video stream or file, loop back to start
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                time.sleep(0.05)
                continue

            frame_count += 1
            now = time.time()
            if now - start_time >= 1.0:
                self.current_fps = frame_count / (now - start_time)
                frame_count = 0
                start_time = now
                ws_manager.sync_broadcast({"type": "STATUS_UPDATE", "status": self.get_status()})

            # Perform Object Detection + Tracking
            persons_count = 0
            if self.model is not None:
                try:
                    results = self.model.track(
                        frame,
                        persist=True,
                        conf=self.confidence_threshold,
                        verbose=False
                    )
                    result = results[0]
                    boxes = result.boxes

                    if boxes is not None and boxes.id is not None:
                        track_ids = boxes.id.int().cpu().tolist()

                        for box, track_id in zip(boxes, track_ids):
                            class_id = int(box.cls[0])
                            confidence = float(box.conf[0])

                            # Only track target classes (0 = person)
                            if class_id in self.target_class_ids:
                                persons_count += 1

                                # Handle new track ID event
                                if track_id not in self.alerted_ids:
                                    self.alerted_ids.add(track_id)
                                    self.total_session_events += 1
                                    
                                    # Handle event in background thread to avoid frame drops
                                    threading.Thread(
                                        target=self._handle_detection_event,
                                        args=(frame.copy(), track_id, "person", confidence),
                                        daemon=True
                                    ).start()

                    # Render bounding boxes onto frame
                    annotated_frame = result.plot()
                except Exception as err:
                    print(f"⚠️ Tracking error: {err}")
                    annotated_frame = frame
            else:
                annotated_frame = frame

            self.active_persons_in_frame = persons_count

            # Add status overlay to top-left of video stream
            overlay_text = f"FPS: {self.current_fps:.1f} | Persons in frame: {persons_count}"
            cv2.putText(annotated_frame, overlay_text, (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

            # Encode frame to JPEG
            ret, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ret:
                with self.frame_lock:
                    self.current_frame_bytes = buffer.tobytes()

            time.sleep(0.01)  # prevent maxing out CPU

        cap.release()
        self.is_running = False

    def _handle_detection_event(self, frame, track_id: int, object_class: str, confidence: float):
        timestamp_sec = int(time.time())
        filename = f"person_{track_id}_{timestamp_sec}.jpg"
        screenshot_path = os.path.join(self.screenshot_dir, filename)

        # Save screenshot file
        cv2.imwrite(screenshot_path, frame)
        print(f"📸 Saved event screenshot: {screenshot_path}")

        # Add event to SQLite Database
        event_id = add_event(
            camera_name=self.camera_name,
            track_id=track_id,
            object_class=object_class,
            confidence=confidence,
            screenshot_filename=filename,
            screenshot_path=screenshot_path
        )

        event_info = {
            "id": event_id,
            "camera_name": self.camera_name,
            "track_id": track_id,
            "object_class": object_class,
            "confidence": round(confidence, 2),
            "filename": filename,
            "time": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        self.latest_event_info = event_info
        self.last_event_time = event_info["time"]

        # Broadcast real-time event via WebSockets
        ws_manager.sync_broadcast({"type": "NEW_EVENT", "event": event_info})

        # Send notification via Discord
        webhook_url = get_setting("DISCORD_WEBHOOK_URL", os.getenv("DISCORD_WEBHOOK_URL", ""))
        send_discord_notification(
            image_path=screenshot_path,
            track_id=track_id,
            camera_name=self.camera_name,
            object_class=object_class,
            confidence=confidence,
            webhook_url=webhook_url
        )

    def process_single_frame(self, image_bytes: bytes, confidence: float = 0.5, camera_name: str = "Mobile Camera") -> Dict[str, Any]:
        """Processes a single frame uploaded directly from HTML5 Mobile/Web Browser Camera."""
        import numpy as np
        import base64

        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return {"error": "Invalid frame image data"}

        persons_count = 0
        annotated_frame = frame
        if self.model is not None:
            try:
                results = self.model.track(
                    frame,
                    persist=True,
                    conf=confidence,
                    verbose=False
                )
                result = results[0]
                boxes = result.boxes

                if boxes is not None and boxes.id is not None:
                    track_ids = boxes.id.int().cpu().tolist()

                    for box, track_id in zip(boxes, track_ids):
                        class_id = int(box.cls[0])
                        conf_score = float(box.conf[0])

                        if class_id in self.target_class_ids:
                            persons_count += 1

                            if track_id not in self.alerted_ids:
                                self.alerted_ids.add(track_id)
                                self.total_session_events += 1
                                threading.Thread(
                                    target=self._handle_detection_event,
                                    args=(frame.copy(), track_id, "person", conf_score),
                                    daemon=True
                                ).start()

                annotated_frame = result.plot()
            except Exception as e:
                print(f"Frame process error: {e}")

        # Encode annotated frame back to base64 JPEG data URL
        ret, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if not ret:
            return {"error": "Encoding failed"}

        encoded_str = base64.b64encode(buffer.tobytes()).decode('utf-8')
        return {
            "status": "success",
            "annotated_image": f"data:image/jpeg;base64,{encoded_str}",
            "persons_count": persons_count
        }

    def get_frame_bytes(self) -> Optional[bytes]:
        with self.frame_lock:
            return self.current_frame_bytes

    def get_status(self) -> Dict[str, Any]:
        return {
            "is_running": self.is_running,
            "camera_source": str(self.camera_source),
            "camera_name": self.camera_name,
            "confidence_threshold": self.confidence_threshold,
            "fps": round(self.current_fps, 1),
            "active_persons": self.active_persons_in_frame,
            "session_events": self.total_session_events,
            "last_event_time": self.last_event_time,
            "latest_event": self.latest_event_info
        }

# Global Instance
camera_manager = CameraManager()
