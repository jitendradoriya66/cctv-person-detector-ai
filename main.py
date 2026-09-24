import os
import time
from typing import Optional, Dict, Any
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from dotenv import load_dotenv

from database import init_db, get_events, delete_event, clear_all_events, get_stats, get_setting, save_setting
from camera_manager import camera_manager
from discord_notifier import send_discord_notification

# Load environment variables
load_dotenv()

# Initialize FastAPI App
app = FastAPI(
    title="AI CCTV Monitoring Sentinel System",
    description="Real-Time CCTV Person Detection, Tracking, SQLite Storage & Discord Alerts",
    version="2.0.0"
)

# Initialize Database
init_db()

# Mount Screenshots Static Directory
SCREENSHOT_DIR = "screenshots"
os.makedirs(SCREENSHOT_DIR, exist_ok=True)
app.mount("/screenshots", StaticFiles(directory=SCREENSHOT_DIR), name="screenshots")

# Setup Jinja2 Templates
templates = Jinja2Templates(directory="templates")


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
# 2. MJPEG LIVE STREAMING ENDPOINT
# ============================================================
def generate_mjpeg_frames():
    """Generator function that yields JPEG frames for MJPEG HTTP streaming."""
    while True:
        frame_bytes = camera_manager.get_frame_bytes()
        if frame_bytes is not None:
            yield (
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n'
            )
        time.sleep(0.03)  # ~30 FPS throttle

@app.get("/video_feed")
async def video_feed():
    """Provides a continuous MJPEG multipart stream for live viewing in <img> tags."""
    if not camera_manager.is_running:
        # Start default camera source if not already running
        camera_manager.start()
        
    return StreamingResponse(
        generate_mjpeg_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


# ============================================================
# 3. CAMERA CONTROL APIs
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
        raise HTTPException(status_code=500, detail="Failed to start camera stream.")

@app.post("/api/stop-camera")
async def stop_camera():
    success = camera_manager.stop()
    if success:
        return {"status": "success", "message": "Camera stopped successfully."}
    else:
        raise HTTPException(status_code=500, detail="Failed to stop camera stream.")


# ============================================================
# 4. SYSTEM STATUS & METRICS API
# ============================================================
@app.get("/api/status")
async def get_system_status():
    status = camera_manager.get_status()
    db_stats = get_stats()
    status["stats"] = db_stats
    return JSONResponse(status)


# ============================================================
# 5. DETECTION EVENTS APIs
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
# 6. SETTINGS & DISCORD TEST APIs
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

    # Find a sample screenshot if available
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