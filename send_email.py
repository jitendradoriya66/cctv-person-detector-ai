import smtplib
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

def send_report_email(
    to_email: str = "jitendra662004@gmail.com",
    sender_email: str = None,
    sender_password: str = None,
    smtp_server: str = "smtp.gmail.com",
    smtp_port: int = 587
):
    """
    Sends the AI CCTV System Development & Architecture Report via SMTP Email.
    """
    sender_email = sender_email or os.getenv("SENDER_EMAIL", "")
    sender_password = sender_password or os.getenv("SENDER_PASSWORD", "")

    if not sender_email or not sender_password:
        print("⚠️ Sender email or password not provided. Please set SENDER_EMAIL and SENDER_PASSWORD env vars or pass credentials.")
        return False

    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = to_email
    msg['Subject'] = "AI CCTV Person Detection System - Complete Architecture & Deployment Report"

    body = """
Hello Jitendra,

Here is the complete development and architectural status report for the AI CCTV Person Detection & Security Monitoring System.

============================================================
SYSTEM ARCHITECTURE & MODULE STATUS (100% COMPLETE)
============================================================

1. CCTV Camera Source: 🟢 Complete
   - Supports local webcams (0, 1), custom RTSP CCTV stream URLs, and video files.

2. Frame Extraction (OpenCV): 🟢 Complete
   - Asynchronous non-blocking frame processor with live FPS counter.

3. Object Detection (YOLO): 🟢 Complete
   - Ultralytics YOLOv8/YOLO11 person detection with real-time bounding boxes and confidence sliders.

4. Object Tracking (ByteTrack / DeepSORT ID): 🟢 Complete
   - Identity tracking across video frames (e.g. ID #17), eliminating duplicate spam alerts.

5. Screenshot & Discord Notifications: 🟢 Complete
   - Instant screenshot captures sent as rich Discord Webhook embeds with attached image files.

6. Database & Storage (SQLite): 🟢 Complete
   - SQLite persistent database storing event history (Timestamp, Camera Name, Track ID, Confidence, Screenshot Path).

7. FastAPI Backend & Responsive Web Dashboard: 🟢 Complete
   - Live MJPEG video stream (/video_feed), REST API endpoints, dark-mode glassmorphism interface fully optimized for both desktop and mobile devices.

============================================================
LIVE CLOUD DEPLOYMENT (RENDER RECOMMENDATION)
============================================================

Can Render work alone for live deployment?
- YES! Render can host the FastAPI Backend, Web Dashboard, Database, and Discord Notification engine.
- Key Requirement for Live Cameras: Render cloud containers run in remote AWS data centers and cannot access a local USB webcam (camera index 0). For live physical CCTV cameras on Render, you must configure an accessible RTSP Stream URL (e.g., rtsp://your-ip:554/stream).
- Disk Persistence: Attach a Render Persistent Disk to preserve SQLite database logs and screenshot images across container restarts.

============================================================
GITHUB REPOSITORY SETUP
============================================================
Recommended Repository Name: cctv-person-detector-ai

Commands to push your code:
1. git init
2. git add .
3. git commit -m "Initial commit: AI CCTV Security System"
4. git branch -M main
5. git remote add origin https://github.com/YOUR_USERNAME/cctv-person-detector-ai.git
6. git push -u origin main

Best regards,
AI CCTV Sentinel System Team
"""

    msg.attach(MIMEText(body, 'plain'))

    try:
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(sender_email, sender_password)
        server.send_message(msg)
        server.quit()
        print(f"✅ Report email successfully sent to {to_email}")
        return True
    except Exception as e:
        print(f"❌ Failed to send email: {e}")
        return False

if __name__ == "__main__":
    send_report_email()
