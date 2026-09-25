import os
os.environ["OPENCV_LOG_LEVEL"] = "OFF"
import cv2
import time
import logging
import threading
import numpy as np
import urllib3
import requests
from typing import Optional, Dict, Any, List
from ultralytics import YOLO
from database import add_event, get_setting, save_setting
from discord_notifier import send_discord_notification
from websocket_manager import ws_manager

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Configure Application Logger
logger = logging.getLogger("cctv_ai.camera_manager")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

DEFAULT_DEMO_URL = "https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/people-detection.mp4"
DEFAULT_DEMO_2_URL = "https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/head-pose-face-detection-female.mp4"

class CameraManager:
    """
    Background Camera & YOLO Object Detection Engine.
    Handles OpenCV RTSP/Webcam streaming, object detection, tracking,
    MJPEG frame generation, database logging, and Discord notifications.
    Fully thread-safe and resilient for cloud (Render) and local environments.
    """
    def __init__(self, model_path: str = "yolo26n.pt", screenshot_dir: str = "screenshots"):
        self.model_path = model_path
        self.screenshot_dir = screenshot_dir
        os.makedirs(self.screenshot_dir, exist_ok=True)

        self.model: Optional[YOLO] = None
        self._model_lock = threading.Lock()
        self._is_model_loading = False

        self.is_running = False
        self.stream_status = "STOPPED"  # STOPPED, STARTING, RUNNING, FAILED
        self.stream_error: Optional[str] = None
        self.camera_thread: Optional[threading.Thread] = None

        # Synchronization Locks
        self.state_lock = threading.Lock()
        self.tracker_lock = threading.Lock()
        self.frame_lock = threading.Lock()

        self.current_frame_bytes: Optional[bytes] = None
        self._create_placeholder_frame("AI CCTV Camera System Ready")

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

    def get_model(self) -> Optional[YOLO]:
        """Lazy thread-safe model loader."""
        if self.model is not None:
            return self.model

        with self._model_lock:
            if self.model is not None:
                return self.model
            try:
                logger.info(f"Loading YOLO model lazily from {self.model_path}...")
                self.model = YOLO(self.model_path)
                logger.info("YOLO Model loaded successfully!")
                return self.model
            except Exception as e:
                logger.error(f"Failed to load YOLO model: {e}", exc_info=True)
                return None

    def reset_tracker(self):
        """Resets YOLO tracker internal state safely across stream changes."""
        with self.tracker_lock:
            if self.model is not None and hasattr(self.model, "predictor") and self.model.predictor is not None:
                try:
                    if hasattr(self.model.predictor, "trackers") and self.model.predictor.trackers:
                        for tracker in self.model.predictor.trackers:
                            if hasattr(tracker, "reset"):
                                tracker.reset()
                    logger.info("YOLO Tracker state reset cleanly.")
                except Exception as err:
                    logger.warning(f"Could not reset tracker state: {err}")

    def start(self, source: Any = None, confidence: float = None, camera_name: str = None) -> bool:
        with self.state_lock:
            if self.is_running:
                logger.info("Dynamic camera configuration update requested. Stopping current stream...")
                self._stop_internal()

            if source is not None:
                if source == "demo1":
                    self.camera_source = DEFAULT_DEMO_URL
                    self.camera_name = "Demo CCTV Stream 1"
                elif source == "demo2":
                    self.camera_source = DEFAULT_DEMO_2_URL
                    self.camera_name = "Demo CCTV Stream 2"
                elif isinstance(source, str) and source.isdigit():
                    self.camera_source = int(source)
                else:
                    self.camera_source = source

            if confidence is not None:
                self.confidence_threshold = float(confidence)

            if camera_name is not None:
                self.camera_name = camera_name

            # Pre-load model before starting stream to verify readiness
            model = self.get_model()
            if model is None:
                self.stream_status = "FAILED"
                self.stream_error = "YOLO Model initialization failed."
                self._create_placeholder_frame("Error: Model Init Failed")
                logger.error("Cannot start stream: YOLO model failed to initialize.")
                return False

            self.reset_tracker()
            self.alerted_ids.clear()
            self.active_persons_in_frame = 0
            self.current_fps = 0.0
            self.stream_error = None
            self.stream_status = "STARTING"
            self.is_running = True

            self.camera_thread = threading.Thread(target=self._process_stream, daemon=True)
            self.camera_thread.start()
            logger.info(f"Camera stream thread launched for source: {self.camera_source}")
            return True

    def stop(self) -> bool:
        with self.state_lock:
            return self._stop_internal()

    def _stop_internal(self) -> bool:
        if not self.is_running and self.stream_status == "STOPPED":
            logger.info("Camera stream is already stopped.")
            return True

        self.is_running = False
        self.stream_status = "STOPPED"
        self.active_persons_in_frame = 0
        self.current_fps = 0.0

        if self.camera_thread and self.camera_thread.is_alive():
            if threading.current_thread() != self.camera_thread:
                self.camera_thread.join(timeout=3.0)

        self._create_placeholder_frame("Camera Stream Stopped")
        logger.info("Camera stream stopped successfully.")
        return True

    def _download_demo_video(self, url: str, local_filename: str) -> Optional[str]:
        """Downloads demo video file locally to bypass SSL / HTTP redirect issues with OpenCV."""
        local_path = os.path.join(self.screenshot_dir, local_filename)
        if os.path.exists(local_path) and os.path.getsize(local_path) > 100000:
            return local_path

        try:
            logger.info(f"Downloading demo video to local cache: {local_path}...")
            response = requests.get(url, stream=True, timeout=3, verify=False)
            if response.status_code == 200:
                with open(local_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=32768):
                        f.write(chunk)
                if os.path.exists(local_path) and os.path.getsize(local_path) > 100000:
                    logger.info(f"Demo video cached successfully: {local_path} ({os.path.getsize(local_path)} bytes)")
                    return local_path
        except Exception as e:
            logger.warning(f"Could not download demo video locally: {e}")

        return None

    def _open_and_validate_source(self, source: Any) -> tuple[Optional[cv2.VideoCapture], bool, Optional[np.ndarray], str]:
        """
        Attempts to open VideoCapture and reads the first test frame.
        Returns (cap, success, first_frame, error_msg)
        """
        try:
            logger.info(f"Attempting to open VideoCapture source: {source}")
            cap = cv2.VideoCapture(source)
            if not cap.isOpened():
                return None, False, None, f"OpenCV VideoCapture failed to open source: {source}"

            ret, frame = cap.read()
            if not ret or frame is None or frame.size == 0:
                cap.release()
                return None, False, None, f"VideoCapture opened source '{source}' but failed to read initial frame."

            h, w = frame.shape[:2]
            logger.info(f"Successfully validated source '{source}'. First frame resolution: {w}x{h}")
            return cap, True, frame, ""
        except Exception as e:
            return None, False, None, f"Exception while opening source '{source}': {str(e)}"

    def _generate_synthetic_cctv_frame(self, t: float) -> np.ndarray:
        """Generates a synthetic CCTV surveillance stream frame for fallback testing."""
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        img[:, :] = (30, 25, 20)
        cv2.rectangle(img, (20, 20), (620, 460), (60, 50, 40), 2)
        
        # Animated simulated person shape
        x = int(120 + 320 * (0.5 + 0.5 * np.sin(t * 0.8)))
        y = int(140 + 100 * (0.5 + 0.5 * np.cos(t * 0.5)))
        
        # Head and body
        cv2.circle(img, (x + 30, y + 20), 16, (180, 180, 180), -1)
        cv2.rectangle(img, (x + 10, y + 36), (x + 50, y + 120), (160, 160, 160), -1)
        
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(img, f"DEMO CCTV SURVEILLANCE FEED | {timestamp}", (30, 440),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 160), 1, cv2.LINE_AA)
        return img

    def _process_stream(self):
        source_target = self.camera_source
        is_cloud_env = (os.name != 'nt')

        # Check Linux cloud environment (/dev/video0 missing)
        if is_cloud_env:
            if isinstance(source_target, int) or (isinstance(source_target, str) and str(source_target).isdigit()):
                dev_path = f"/dev/video{source_target}"
                if not os.path.exists(dev_path):
                    logger.info(f"Cloud container detected without '{dev_path}'. Switching to online CCTV demo video...")
                    source_target = DEFAULT_DEMO_URL
                    self.camera_name = "Demo CCTV Stream"

        # If source is a remote demo URL, try local cached version first
        if source_target == DEFAULT_DEMO_URL:
            local_demo = self._download_demo_video(DEFAULT_DEMO_URL, "demo_people_detection.mp4")
            if local_demo:
                source_target = local_demo
        elif source_target == DEFAULT_DEMO_2_URL:
            local_demo = self._download_demo_video(DEFAULT_DEMO_2_URL, "demo_female_face.mp4")
            if local_demo:
                source_target = local_demo

        # Try primary source validation
        cap, success, first_frame, err_msg = self._open_and_validate_source(source_target)

        # Fallback handling if primary source failed
        if not success:
            logger.warning(f"Primary source '{source_target}' failed validation: {err_msg}")
            
            # Try fallback demo video if primary was not already fallback
            if source_target != DEFAULT_DEMO_URL:
                logger.info("Attempting automatic fallback to cached demo video...")
                fallback_path = self._download_demo_video(DEFAULT_DEMO_URL, "demo_people_detection.mp4") or DEFAULT_DEMO_URL
                cap, success, first_frame, fallback_err = self._open_and_validate_source(fallback_path)
                if success:
                    self.camera_name = "Demo CCTV Stream (Fallback)"
                    logger.info("Successfully switched to fallback video stream.")

        use_synthetic_fallback = False
        if not success or cap is None:
            logger.info("OpenCV VideoCapture unavailable. Activating synthetic CCTV demo stream...")
            use_synthetic_fallback = True

        with self.state_lock:
            self.stream_status = "RUNNING"
            self.stream_error = None

        ws_manager.sync_broadcast({"type": "STATUS_UPDATE", "status": self.get_status()})

        frame_count = 0
        start_time = time.time()
        consecutive_read_failures = 0
        current_frame = first_frame if not use_synthetic_fallback else None
        synthetic_start_t = time.time()

        try:
            while self.is_running:
                if use_synthetic_fallback:
                    frame = self._generate_synthetic_cctv_frame(time.time() - synthetic_start_t)
                else:
                    if current_frame is None:
                        ret, frame = cap.read()
                        if not ret or frame is None:
                            consecutive_read_failures += 1
                            # Attempt 1: Seek to frame 0
                            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            ret, frame = cap.read()
                            if ret and frame is not None and frame.size > 0:
                                consecutive_read_failures = 0
                            else:
                                # Attempt 2: Re-open VideoCapture handle from source
                                logger.info("Video end or read failure. Re-opening VideoCapture source to loop...")
                                cap.release()
                                cap = cv2.VideoCapture(source_target)
                                ret, frame = cap.read()
                                if ret and frame is not None and frame.size > 0:
                                    consecutive_read_failures = 0
                                else:
                                    if consecutive_read_failures >= 5:
                                        logger.warning("VideoCapture stream unrecoverable. Switching to continuous synthetic CCTV stream...")
                                        use_synthetic_fallback = True
                                        consecutive_read_failures = 0
                                        continue
                                    time.sleep(0.1)
                                    continue
                    else:
                        frame = current_frame
                        current_frame = None

                consecutive_read_failures = 0
                frame_count += 1
                now = time.time()
                if now - start_time >= 1.0:
                    self.current_fps = frame_count / (now - start_time)
                    frame_count = 0
                    start_time = now
                    ws_manager.sync_broadcast({"type": "STATUS_UPDATE", "status": self.get_status()})

                # Perform Object Detection + Tracking under tracker_lock
                persons_count = 0
                annotated_frame = frame
                model = self.get_model()

                if model is not None:
                    try:
                        with self.tracker_lock:
                            results = model.track(
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

                                if class_id in self.target_class_ids:
                                    persons_count += 1

                                    if track_id not in self.alerted_ids:
                                        self.alerted_ids.add(track_id)
                                        self.total_session_events += 1
                                        threading.Thread(
                                            target=self._handle_detection_event,
                                            args=(frame.copy(), track_id, "person", confidence),
                                            daemon=True
                                        ).start()

                        annotated_frame = result.plot()
                    except Exception as err:
                        logger.error(f"Tracking error during stream processing: {err}", exc_info=True)
                        annotated_frame = frame

                self.active_persons_in_frame = persons_count

                # Add status overlay
                overlay_text = f"FPS: {self.current_fps:.1f} | Persons in frame: {persons_count}"
                cv2.putText(annotated_frame, overlay_text, (15, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

                # Encode frame to JPEG
                ret, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ret:
                    with self.frame_lock:
                        self.current_frame_bytes = buffer.tobytes()

                time.sleep(0.01)

        except Exception as ex:
            logger.error(f"Unhandled exception in stream loop: {ex}", exc_info=True)
            with self.state_lock:
                self.stream_status = "FAILED"
                self.stream_error = str(ex)
        finally:
            if cap is not None:
                cap.release()
                logger.info("VideoCapture released cleanly.")
            with self.state_lock:
                self.is_running = False
                if self.stream_status == "RUNNING":
                    self.stream_status = "STOPPED"
                self.active_persons_in_frame = 0

            if self.stream_status == "FAILED":
                self._create_placeholder_frame(f"Stream Failed: {self.stream_error or 'Unknown Error'}")
            else:
                self._create_placeholder_frame("Camera Stream Stopped")

            ws_manager.sync_broadcast({"type": "STATUS_UPDATE", "status": self.get_status()})

    def _handle_detection_event(self, frame: np.ndarray, track_id: int, object_class: str, confidence: float):
        try:
            timestamp_sec = int(time.time())
            filename = f"person_{track_id}_{timestamp_sec}.jpg"
            screenshot_path = os.path.join(self.screenshot_dir, filename)

            cv2.imwrite(screenshot_path, frame)
            logger.info(f"Saved event screenshot: {screenshot_path}")

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

            ws_manager.sync_broadcast({"type": "NEW_EVENT", "event": event_info})

            webhook_url = get_setting("DISCORD_WEBHOOK_URL", os.getenv("DISCORD_WEBHOOK_URL", ""))
            send_discord_notification(
                image_path=screenshot_path,
                track_id=track_id,
                camera_name=self.camera_name,
                object_class=object_class,
                confidence=confidence,
                webhook_url=webhook_url
            )
        except Exception as e:
            logger.error(f"Error handling detection event for track {track_id}: {e}", exc_info=True)

    def process_single_frame(self, image_bytes: bytes, confidence: float = 0.5, camera_name: str = "Mobile Camera") -> Dict[str, Any]:
        """
        Processes a single frame uploaded from HTML5 Mobile/Web Browser Camera.
        Thread-safe, offloaded, and synchronized with persistent tracker state lock.
        """
        import base64

        if not image_bytes or len(image_bytes) == 0:
            return {"status": "error", "error": "Empty frame data received"}

        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None or frame.size == 0:
            return {"status": "error", "error": "Failed to decode frame image"}

        # Resize very large mobile images to optimize CPU on Render
        h, w = frame.shape[:2]
        if w > 1280 or h > 720:
            scale = min(1280.0 / w, 720.0 / h)
            new_w, new_h = int(w * scale), int(h * scale)
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

        model = self.get_model()
        if model is None:
            return {"status": "error", "error": "YOLO model not initialized"}

        persons_count = 0
        annotated_frame = frame

        try:
            # Synchronize model tracking call across threads
            with self.tracker_lock:
                results = model.track(
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
            logger.error(f"Single frame processing error: {e}", exc_info=True)
            return {"status": "error", "error": str(e)}

        self.active_persons_in_frame = persons_count

        ret, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if not ret:
            return {"status": "error", "error": "Encoding annotated frame failed"}

        encoded_str = base64.b64encode(buffer.tobytes()).decode('utf-8')
        return {
            "status": "success",
            "annotated_image": f"data:image/jpeg;base64,{encoded_str}",
            "persons_count": persons_count
        }

    def _create_placeholder_frame(self, text: str = "AI CCTV Camera System Ready"):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        img[:, :] = (25, 20, 15)
        # Wrap long text
        lines = [text[i:i+35] for i in range(0, len(text), 35)]
        y0 = 220 if len(lines) == 1 else 200
        for i, line in enumerate(lines):
            cv2.putText(img, line, (40, y0 + (i * 30)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 160), 2, cv2.LINE_AA)
        ret, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ret:
            with self.frame_lock:
                self.current_frame_bytes = buffer.tobytes()

    def get_frame_bytes(self) -> Optional[bytes]:
        with self.frame_lock:
            if self.current_frame_bytes is None:
                self._create_placeholder_frame("AI CCTV Camera System Ready")
            return self.current_frame_bytes

    def get_status(self) -> Dict[str, Any]:
        return {
            "is_running": self.is_running,
            "stream_status": self.stream_status,
            "stream_error": self.stream_error,
            "camera_source": str(self.camera_source),
            "camera_name": self.camera_name,
            "confidence_threshold": self.confidence_threshold,
            "fps": round(self.current_fps, 1),
            "active_persons": self.active_persons_in_frame,
            "session_events": self.total_session_events,
            "last_event_time": self.last_event_time,
            "latest_event": self.latest_event_info
        }

# Global Singleton Instance
camera_manager = CameraManager()
