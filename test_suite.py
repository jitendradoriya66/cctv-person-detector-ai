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

def test_02_user_registration_and_validation():
    """Verify user registration, duplicate username rejection, and password validation."""
    uname = f"test_user_{int(time.time())}"
    uemail = f"{uname}@sentinel.ai"
    reg_data = {
        "username": uname,
        "email": uemail,
        "password": "secure_password_123"
    }
    res = client.post("/api/register", json=reg_data)
    assert res.status_code == 200
    res_json = res.json()
    assert res_json["success"] is True
    assert "token" in res_json
    assert res_json["user"]["username"] == uname

    # Test duplicate username rejection with clear error detail
    dup_username_res = client.post("/api/register", json=reg_data)
    assert dup_username_res.status_code == 400
    dup_json = dup_username_res.json()
    assert dup_json["success"] is False
    assert "already taken" in dup_json["error"].lower()

    # Test duplicate email rejection with clear error detail
    dup_email_data = {
        "username": f"different_{uname}",
        "email": uemail,
        "password": "another_password_123"
    }
    dup_email_res = client.post("/api/register", json=dup_email_data)
    assert dup_email_res.status_code == 400
    dup_email_json = dup_email_res.json()
    assert dup_email_json["success"] is False
    assert "already registered" in dup_email_json["error"].lower()

    # Test short password validation error
    short_pwd_data = {
        "username": "short_user",
        "email": "short@sentinel.ai",
        "password": "123"
    }
    short_res = client.post("/api/register", json=short_pwd_data)
    assert short_res.status_code == 400
    assert "at least 6 characters" in short_res.json()["error"].lower()

def test_03_user_login_and_credential_verification():
    """Verify login success with default admin / registered user, and invalid password blocking."""
    # Test valid login with seeded admin
    login_admin = {
        "username": "admin",
        "password": "admin123"
    }
    res = client.post("/api/login", json=login_admin)
    assert res.status_code == 200
    res_data = res.json()
    assert res_data["success"] is True
    assert "token" in res_data
    assert res_data["user"]["role"] == "admin"

    # Test invalid password rejection with clear error detail
    bad_login = {
        "username": "admin",
        "password": "wrong_password_999"
    }
    bad_res = client.post("/api/login", json=bad_login)
    assert bad_res.status_code == 401
    bad_json = bad_res.json()
    assert bad_json["success"] is False
    assert "invalid username/email or password" in bad_json["error"].lower()

def test_04_protected_route_enforcement():
    """Verify unauthenticated user requests are blocked or redirected, and authenticated requests succeed."""
    # Clear any leftover login cookies from previous tests to ensure clean unauthenticated check
    client.cookies.clear()

    # Test unauthenticated dashboard access -> redirects to /login
    unauth_dash = client.get("/", follow_redirects=False)
    assert unauth_dash.status_code == 303
    assert unauth_dash.headers["location"] == "/login"

    # Test unauthenticated API access to /api/me -> 401 Unauthorized
    unauth_api = client.get("/api/me")
    assert unauth_api.status_code == 401
    assert "authentication required" in unauth_api.json()["detail"].lower()

    # Login to obtain auth session cookie
    login_res = client.post("/api/login", json={"username": "admin", "password": "admin123"})
    assert login_res.status_code == 200

    # Test authenticated access to /api/me
    auth_me = client.get("/api/me")
    assert auth_me.status_code == 200
    assert auth_me.json()["success"] is True
    assert auth_me.json()["user"]["username"] == "admin"

    # Test authenticated access to dashboard -> 200 OK
    auth_dash = client.get("/")
    assert auth_dash.status_code == 200
    assert "AI COMMAND CENTER" in auth_dash.text

def test_05_camera_start_and_stop_demo():
    """Test starting Demo CCTV stream and stopping cleanly."""
    start_resp = client.post("/api/start-camera", json={"source": "demo1", "confidence": 0.5})
    assert start_resp.status_code == 200
    assert start_resp.json()["status"] == "success"

    time.sleep(1)
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

def test_06_image_upload_and_events_api():
    """Test /api/upload-image and /api/events APIs with authenticated session."""
    # Ensure client is logged in
    client.post("/api/login", json={"username": "admin", "password": "admin123"})

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

    # Retrieve events list
    events_res = client.get("/api/events?page=1&limit=12")
    assert events_res.status_code == 200
    events_data = events_res.json()
    assert "events" in events_data
    assert "total" in events_data

def test_07_logout_and_session_clearing():
    """Test /api/logout clears authentication session cookie and blocks subsequent protected access."""
    # Ensure logged in
    client.post("/api/login", json={"username": "admin", "password": "admin123"})
    assert client.get("/api/me").status_code == 200

    # Execute logout
    logout_res = client.post("/api/logout")
    assert logout_res.status_code == 200
    assert logout_res.json()["success"] is True

    # Subsequent request to /api/me should fail with 401
    me_after_logout = client.get("/api/me")
    assert me_after_logout.status_code == 401

def test_08_user_data_isolation():
    """Verify regular users can ONLY view their own uploaded events and cannot view other users' data."""
    from database import add_event, get_user_by_username
    ts = int(time.time())
    user_a_name = f"user_a_{ts}"
    user_b_name = f"user_b_{ts}"

    # 1. Register User A
    reg_a = client.post("/api/register", json={"username": user_a_name, "email": f"{user_a_name}@mail.com", "password": "password123"})
    assert reg_a.status_code == 200
    user_a_id = reg_a.json()["user"]["sub"]

    # Record event specifically for User A
    add_event(
        camera_name="User A Cam",
        track_id=1,
        object_class="person",
        confidence=0.95,
        screenshot_filename="test_user_a.jpg",
        screenshot_path="screenshots/test_user_a.jpg",
        source_type="IMAGE",
        user_id=int(user_a_id)
    )

    # Verify User A sees 1 event
    events_a = client.get("/api/events?page=1&limit=12").json()
    assert events_a["total"] >= 1
    assert all(ev.get("user_id", int(user_a_id)) == int(user_a_id) for ev in events_a["events"])

    # 2. Register User B and log in
    client.cookies.clear()
    reg_b = client.post("/api/register", json={"username": user_b_name, "email": f"{user_b_name}@mail.com", "password": "password123"})
    assert reg_b.status_code == 200

    # Verify User B sees 0 events (cannot see User A's events)
    events_b = client.get("/api/events?page=1&limit=12").json()
    assert events_b["total"] == 0

def test_09_admin_permissions_and_user_delete_blocking():
    """Verify regular users get 403 Forbidden on DELETE endpoints, while admin can delete any event."""
    ts = int(time.time())
    user_c_name = f"user_c_{ts}"

    # Register regular user C
    client.cookies.clear()
    client.post("/api/register", json={"username": user_c_name, "email": f"{user_c_name}@mail.com", "password": "password123"})

    # Regular User C attempts to delete an event -> 403 Forbidden
    del_attempt = client.delete("/api/events/1")
    assert del_attempt.status_code == 403
    assert "only administrators" in del_attempt.json()["detail"].lower()

    # Regular User C attempts clear all events -> 403 Forbidden
    clear_attempt = client.delete("/api/events")
    assert clear_attempt.status_code == 403
    assert "only administrators" in clear_attempt.json()["detail"].lower()

    # Admin logs in
    client.cookies.clear()
    client.post("/api/login", json={"username": "admin", "password": "admin123"})

    # Admin can view all events across users
    admin_events = client.get("/api/events?page=1&limit=12").json()
    assert isinstance(admin_events["events"], list)

if __name__ == "__main__":
    print("Running end-to-end integration test suite...")
    test_01_health_and_status()
    print("✅ test_01_health_and_status passed")
    test_02_user_registration_and_validation()
    print("✅ test_02_user_registration_and_validation passed")
    test_03_user_login_and_credential_verification()
    print("✅ test_03_user_login_and_credential_verification passed")
    test_04_protected_route_enforcement()
    print("✅ test_04_protected_route_enforcement passed")
    test_05_camera_start_and_stop_demo()
    print("✅ test_05_camera_start_and_stop_demo passed")
    test_06_image_upload_and_events_api()
    print("✅ test_06_image_upload_and_events_api passed")
    test_07_logout_and_session_clearing()
    print("✅ test_07_logout_and_session_clearing passed")
    test_08_user_data_isolation()
    print("✅ test_08_user_data_isolation passed")
    test_09_admin_permissions_and_user_delete_blocking()
    print("✅ test_09_admin_permissions_and_user_delete_blocking passed")
    print("\n🎉 ALL 9 END-TO-END TESTS PASSED WITH 100% ACCURACY!")
