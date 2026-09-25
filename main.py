import os
import time
import asyncio
import logging
from typing import Optional, Dict, Any
from fastapi import FastAPI, Request, HTTPException, WebSocket, WebSocketDisconnect, status, UploadFile, File, Form
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
from dotenv import load_dotenv

from database import init_db, get_events, delete_event, clear_all_events, get_stats, get_setting, save_setting
from camera_manager import camera_manager
from discord_notifier import send_discord_notification
from websocket_manager import ws_manager

# Configure Main Application Logger
logger = logging.getLogger("cctv_ai.main")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# Load environment variables
load_dotenv()

# Initialize FastAPI App
app = FastAPI(
    title="AI CCTV Monitoring Sentinel System",
    description="Real-Time CCTV Person Detection, Tracking, WebSockets, Redirects & Discord Alerts",
    version="2.1.0"
)

# Mount Screenshots Static Directory
SCREENSHOT_DIR = "screenshots"
os.makedirs(SCREENSHOT_DIR, exist_ok=True)
app.mount("/screenshots", StaticFiles(directory=SCREENSHOT_DIR), name="screenshots")

# Setup Jinja2 Templates
templates = Jinja2Templates(directory="templates")

# FastAPI Application Lifecycle
@app.on_event("startup")
async def startup_event():
    """Initializes database and pre-warms YOLO model safely on application launch."""
    logger.info("Starting AI CCTV Monitoring Sentinel System...")
    init_db()
    # Pre-warm YOLO model in worker thread
    await run_in_threadpool(camera_manager.get_model)
    logger.info("System startup complete and ready for streaming.")


# Request Pydantic Schemas
class StartCameraRequest(BaseModel):
    source: Any = 0
    confidence: Optional[float] = 0.50
    camera_name: Optional[str] = "Entrance Camera"

class SettingsRequest(BaseModel):
    discord_webhook_url: Optional[str] = None
    camera_name: Optional[str] = None


# ============================================================
# 1. WEB DASHBOARD ROUTE
# ============================================================
@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


# ============================================================
# 2. WEBSOCKET REAL-TIME PUSH ENDPOINT
# ============================================================
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    Bi-directional WebSocket connection for instant push updates:
    - Real-time detection alerts & event image prepending
    - Live system status, FPS & person counts without HTTP polling delay
    """
    ws_manager.set_loop(asyncio.get_running_loop())
    await ws_manager.connect(websocket)
    try:
        # Initial status push on connect wrapped safely
        try:
            status_data = camera_manager.get_status()
            status_data["stats"] = get_stats()
            await websocket.send_json({"type": "STATUS_UPDATE", "status": status_data})
        except Exception as push_err:
            logger.warning(f"Initial WebSocket status push error: {push_err}")

        while True:
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
    except Exception as e:
        logger.warning(f"WebSocket client disconnected with error: {e}")
        ws_manager.disconnect(websocket)


# ============================================================
# 3. REDIRECT ROUTES
# ============================================================
@app.get("/redirect/event/{event_id}")
async def redirect_to_event(event_id: int):
    """Redirects client directly to dashboard with target event highlighted."""
    return RedirectResponse(url=f"/?highlight_event={event_id}", status_code=status.HTTP_303_SEE_OTHER)

@app.get("/redirect/start")
async def redirect_start_camera():
    """Starts camera and redirects back to dashboard home."""
    camera_manager.start()
    return RedirectResponse(url="/", status_code=status.HTTP_303_SEE_OTHER)

@app.get("/redirect/stop")
async def redirect_stop_camera():
    """Stops camera and redirects back to dashboard home."""
    camera_manager.stop()
    return RedirectResponse(url="/", status_code=status.HTTP_303_SEE_OTHER)


# ============================================================
# 4. MJPEG LIVE STREAMING ENDPOINT
# ============================================================
def generate_mjpeg_frames():
    """Generator function yielding JPEG frames for MJPEG HTTP streaming."""
    consecutive_stopped = 0
    while True:
        status_info = camera_manager.get_status()
        if not status_info["is_running"] and status_info["stream_status"] in ["STOPPED", "FAILED"]:
            consecutive_stopped += 1
            frame_bytes = camera_manager.get_frame_bytes()
            if frame_bytes is not None:
                yield (
                    b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n'
                )
            # Break stream response after holding placeholder for ~3 seconds
            if consecutive_stopped > 30:
                break
            time.sleep(0.1)
            continue

        consecutive_stopped = 0
        frame_bytes = camera_manager.get_frame_bytes()
        if frame_bytes is not None:
            yield (
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n'
            )
        time.sleep(0.03)  # ~30 FPS throttle

@app.get("/video_feed")
async def video_feed():
    """Provides continuous MJPEG stream for live video viewing."""
    status_info = camera_manager.get_status()
    if not status_info["is_running"] and status_info["stream_status"] not in ["STARTING", "RUNNING"]:
        camera_manager.start()

    return StreamingResponse(
        generate_mjpeg_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


# ============================================================
# 5. CAMERA CONTROL APIs
# ============================================================
@app.post("/api/start-camera")
async def start_camera(req: StartCameraRequest):
    success = camera_manager.start(
        source=req.source,
        confidence=req.confidence,
        camera_name=req.camera_name
    )
    if success:
        return {"status": "success", "message": f"Camera started on source: {req.source}"}
    else:
        status_info = camera_manager.get_status()
        err_detail = status_info.get("stream_error") or "Failed to start camera stream."
        raise HTTPException(status_code=500, detail=err_detail)

@app.post("/api/stop-camera")
async def stop_camera():
    success = camera_manager.stop()
    if success:
        return {"status": "success", "message": "Camera stopped successfully."}
    else:
        raise HTTPException(status_code=500, detail="Failed to stop camera stream.")

@app.post("/api/process-frame")
async def process_browser_frame(
    file: UploadFile = File(...),
    confidence: float = Form(0.5),
    camera_name: str = Form("Smartphone Camera")
):
    """API Endpoint receiving frame uploads from Mobile HTML5 browser camera. Synchronously offloaded to threadpool."""
    contents = await file.read()
    if not contents or len(contents) == 0:
        raise HTTPException(status_code=400, detail="Empty frame uploaded.")

    result = await run_in_threadpool(
        camera_manager.process_single_frame,
        contents,
        confidence=confidence,
        camera_name=camera_name
    )
    return JSONResponse(result)


# ============================================================
# 6. SYSTEM STATUS & METRICS API
# ============================================================
@app.get("/api/status")
async def get_system_status():
    status_data = camera_manager.get_status()
    db_stats = get_stats()
    status_data["stats"] = db_stats
    return JSONResponse(status_data)


# ============================================================
# 7. DETECTION EVENTS APIs
# ============================================================
@app.get("/api/events")
async def list_events(limit: int = 50, offset: int = 0):
    events = get_events(limit=limit, offset=offset)
    return JSONResponse(events)

@app.delete("/api/events/{event_id}")
async def delete_single_event(event_id: int):
    success = delete_event(event_id)
    if success:
        return {"status": "success", "message": f"Event {event_id} deleted."}
    raise HTTPException(status_code=404, detail="Event not found.")

@app.delete("/api/events")
async def clear_events():
    clear_all_events()
    return {"status": "success", "message": "All detection events cleared."}


# ============================================================
# 8. SETTINGS & DISCORD TEST APIs
# ============================================================
@app.post("/api/settings")
async def save_system_settings(req: SettingsRequest):
    if req.discord_webhook_url is not None:
        save_setting("DISCORD_WEBHOOK_URL", req.discord_webhook_url)
    if req.camera_name is not None:
        save_setting("CAMERA_NAME", req.camera_name)
    return {"status": "success", "message": "Settings updated."}

@app.post("/api/test-discord")
async def test_discord_notification():
    webhook_url = get_setting("DISCORD_WEBHOOK_URL", os.getenv("DISCORD_WEBHOOK_URL", ""))
    if not webhook_url:
        return {"status": "error", "message": "Discord Webhook URL is not configured in settings or .env!"}

    screenshots = [f for f in os.listdir(SCREENSHOT_DIR) if f.endswith(".jpg")]
    if screenshots:
        sample_path = os.path.join(SCREENSHOT_DIR, screenshots[0])
    else:
        return {"status": "error", "message": "No screenshot available for test alert."}

    sent = send_discord_notification(
        image_path=sample_path,
        track_id=99,
        camera_name="Test Entrance Camera",
        object_class="person",
        confidence=0.98,
        webhook_url=webhook_url
    )

    if sent:
        return {"status": "success", "message": "Test Discord alert sent successfully!"}
    else:
        return {"status": "error", "message": "Failed to send Discord alert. Check URL or network."}


# Run directly using uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)