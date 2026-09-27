import io
import time
import cv2
import numpy as np
from fastapi.testclient import TestClient
from main import app
from camera_manager import camera_manager

client = TestClient(app)

def test_01_health_and_status():
    """Verify application status endpoint returns expected structure."""
    response = client.get("/api/status")
    assert response.status_code == 200
    data = response.json()
    assert "is_running" in data
    assert "stream_status" in data
    assert "active_persons" in data
    assert "stats" in data
    assert "persons_count" in data["stats"]

def test_02_model_initialization():
    """Verify YOLO model loads lazily and is ready."""
    model = camera_manager.get_model()
    assert model is not None

def test_03_camera_start_and_stop_demo():
    """Test starting Demo CCTV stream and stopping cleanly."""
    start_resp = client.post("/api/start-camera", json={"source": "demo1", "confidence": 0.5})
    assert start_resp.status_code == 200
    assert start_resp.json()["status"] == "success"

    time.sleep(2)
    status_resp = client.get("/api/status")
    assert status_resp.status_code == 200
    status_data = status_resp.json()
    assert status_data["is_running"] is True

    stop_resp = client.post("/api/stop-camera")
    assert stop_resp.status_code == 200
    assert stop_resp.json()["status"] == "success"

    time.sleep(1)
    status_final = client.get("/api/status").json()
    assert status_final["is_running"] is False
    assert status_final["stream_status"] == "STOPPED"

def test_04_process_frame_endpoint():
    """Test /api/process-frame endpoint with synthesized JPEG image."""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(img, "Test Frame", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    _, buffer = cv2.imencode(".jpg", img)
    
    files = {"file": ("test.jpg", io.BytesIO(buffer.tobytes()), "image/jpeg")}
    data = {"confidence": "0.5", "camera_name": "Test Mobile Cam"}

    response = client.post("/api/process-frame", files=files, data=data)
    assert response.status_code == 200
    res_json = response.json()
    assert res_json["status"] == "success"
    assert "annotated_image" in res_json

def test_05_upload_image_endpoint():
    """Test /api/upload-image endpoint for static image YOLO detection."""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.circle(img, (320, 240), 50, (255, 255, 255), -1)
    _, buffer = cv2.imencode(".jpg", img)

    files = {"file": ("sample.jpg", io.BytesIO(buffer.tobytes()), "image/jpeg")}
    data = {"confidence": "0.5", "camera_name": "Test Image Upload"}

    response = client.post("/api/upload-image", files=files, data=data)
    assert response.status_code == 200
    res = response.json()
    assert res["success"] is True
    assert res["source_type"] == "IMAGE"
    assert "annotated_image_url" in res

def test_06_events_api():
    """Test retrieving events list via /api/events with pagination."""
    response = client.get("/api/events?page=1&limit=12")
    assert response.status_code == 200
    res_data = response.json()
    assert isinstance(res_data, dict)
    assert "events" in res_data
    assert "total" in res_data
    assert "total_pages" in res_data
    assert isinstance(res_data["events"], list)

if __name__ == "__main__":
    print("Running integration test suite...")
    test_01_health_and_status()
    print("✅ test_01_health_and_status passed")
    test_02_model_initialization()
    print("✅ test_02_model_initialization passed")
    test_03_camera_start_and_stop_demo()
    print("✅ test_03_camera_start_and_stop_demo passed")
    test_04_process_frame_endpoint()
    print("✅ test_04_process_frame_endpoint passed")
    test_05_upload_image_endpoint()
    print("✅ test_05_upload_image_endpoint passed")
    test_06_events_api()
    print("✅ test_06_events_api passed")
    print("\n🎉 ALL TESTS PASSED SUCCESSFULLY!")
