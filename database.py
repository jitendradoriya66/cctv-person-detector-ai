import sqlite3
import os
import time
from datetime import datetime
from typing import List, Dict, Any, Optional

DB_PATH = "cctv_events.db"

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Events table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            datetime_str TEXT NOT NULL,
            camera_name TEXT NOT NULL,
            track_id INTEGER,
            object_class TEXT NOT NULL,
            confidence REAL NOT NULL,
            screenshot_filename TEXT NOT NULL,
            screenshot_path TEXT NOT NULL
        )
    """)
    
    # Settings table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
    """)
    
    conn.commit()
    conn.close()

def add_event(camera_name: str, track_id: int, object_class: str, confidence: float, screenshot_filename: str, screenshot_path: str) -> int:
    conn = get_db_connection()
    cursor = conn.cursor()
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    cursor.execute("""
        INSERT INTO events (datetime_str, camera_name, track_id, object_class, confidence, screenshot_filename, screenshot_path)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (now_str, camera_name, track_id, object_class, round(confidence, 2), screenshot_filename, screenshot_path))
    
    event_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return event_id

def get_events(limit: int = 50, offset: int = 0, camera_name: Optional[str] = None, object_class: Optional[str] = None) -> List[Dict[str, Any]]:
    conn = get_db_connection()
    cursor = conn.cursor()
    
    query = "SELECT * FROM events"
    params = []
    conditions = []
    
    if camera_name:
        conditions.append("camera_name = ?")
        params.append(camera_name)
    if object_class:
        conditions.append("object_class = ?")
        params.append(object_class)
        
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
        
    query += " ORDER BY id DESC LIMIT ? OFFSET ?"
    params.extend([limit, offset])
    
    cursor.execute(query, params)
    rows = cursor.fetchall()
    
    events = [dict(row) for row in rows]
    conn.close()
    return events

def delete_event(event_id: int) -> bool:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM events WHERE id = ?", (event_id,))
    affected = cursor.rowcount > 0
    conn.commit()
    conn.close()
    return affected

def clear_all_events() -> bool:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM events")
    conn.commit()
    conn.close()
    return True

def get_stats() -> Dict[str, Any]:
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Total events
    cursor.execute("SELECT COUNT(*) FROM events")
    total_events = cursor.fetchone()[0]
    
    # Today's events
    today_str = datetime.now().strftime("%Y-%m-%d")
    cursor.execute("SELECT COUNT(*) FROM events WHERE datetime_str LIKE ?", (f"{today_str}%",))
    today_events = cursor.fetchone()[0]
    
    # Last detection time
    cursor.execute("SELECT datetime_str FROM events ORDER BY id DESC LIMIT 1")
    last_row = cursor.fetchone()
    last_detection = last_row[0] if last_row else "None"
    
    # Persons count
    cursor.execute("SELECT COUNT(*) FROM events WHERE object_class = 'person'")
    persons_count = cursor.fetchone()[0]
    
    conn.close()
    return {
        "total_events": total_events,
        "today_events": today_events,
        "persons_count": persons_count,
        "last_detection": last_detection
    }

def get_setting(key: str, default: str = "") -> str:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT value FROM settings WHERE key = ?", (key,))
    row = cursor.fetchone()
    conn.close()
    return row[0] if row else default

def save_setting(key: str, value: str):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)", (key, value))
    conn.commit()
    conn.close()

def seed_existing_screenshots():
    """Seeds the DB with existing screenshot files if DB is empty."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM events")
    count = cursor.fetchone()[0]
    
    if count == 0 and os.path.exists("screenshots"):
        screenshots = [f for f in os.listdir("screenshots") if f.endswith(".jpg")]
        for fn in sorted(screenshots):
            # Parse track_id if filename follows person_ID_timestamp.jpg format
            parts = fn.replace(".jpg", "").split("_")
            track_id = int(parts[1]) if len(parts) >= 2 and parts[1].isdigit() else 1
            timestamp_sec = int(parts[2]) if len(parts) >= 3 and parts[2].isdigit() else int(time.time())
            date_str = datetime.fromtimestamp(timestamp_sec).strftime("%Y-%m-%d %H:%M:%S")
            
            cursor.execute("""
                INSERT INTO events (datetime_str, camera_name, track_id, object_class, confidence, screenshot_filename, screenshot_path)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (date_str, "Entrance Camera", track_id, "person", 0.92, fn, os.path.join("screenshots", fn)))
        conn.commit()
    conn.close()

# Initialize DB when module loaded
init_db()
seed_existing_screenshots()

