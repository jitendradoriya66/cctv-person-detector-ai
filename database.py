import sqlite3
import os
import time
import math
from datetime import datetime
from typing import List, Dict, Any, Optional

from redis_cache import redis_cache

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
            screenshot_path TEXT NOT NULL,
            source_type TEXT DEFAULT 'STREAM'
        )
    """)
    
    # Check if source_type and user_id columns exist for existing database upgrades
    cursor.execute("PRAGMA table_info(events)")
    columns = [col[1] for col in cursor.fetchall()]
    if "source_type" not in columns:
        try:
            cursor.execute("ALTER TABLE events ADD COLUMN source_type TEXT DEFAULT 'STREAM'")
        except Exception:
            pass
    if "user_id" not in columns:
        try:
            cursor.execute("ALTER TABLE events ADD COLUMN user_id INTEGER DEFAULT 1")
        except Exception:
            pass

    # Settings table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
    """)

    # Users table for Authentication & Authorization
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT DEFAULT 'user',
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    conn.commit()

    # Seed default admin user if no users exist
    cursor.execute("SELECT COUNT(*) FROM users")
    if cursor.fetchone()[0] == 0:
        from auth import hash_password
        default_admin_hash = hash_password("admin123")
        cursor.execute(
            "INSERT INTO users (username, email, password_hash, role) VALUES (?, ?, ?, ?)",
            ("admin", "admin@sentinel.ai", default_admin_hash, "admin")
        )
        conn.commit()

    conn.close()

def create_user(username: str, email: str, password_hash: str, role: str = "user") -> int:
    """Creates a new user record in SQLite database."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO users (username, email, password_hash, role) VALUES (?, ?, ?, ?)",
        (username.strip(), email.strip().lower(), password_hash, role)
    )
    user_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return user_id

def get_user_by_username(username: str) -> Optional[Dict[str, Any]]:
    """Retrieves user record by username."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE LOWER(username) = LOWER(?)", (username.strip(),))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None

def get_user_by_email(email: str) -> Optional[Dict[str, Any]]:
    """Retrieves user record by email."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE LOWER(email) = LOWER(?)", (email.strip(),))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None

def get_user_by_id(user_id: int) -> Optional[Dict[str, Any]]:
    """Retrieves user record by user ID."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None

def add_event(
    camera_name: str,
    track_id: int,
    object_class: str,
    confidence: float,
    screenshot_filename: str,
    screenshot_path: str,
    source_type: str = "STREAM",
    user_id: Optional[int] = 1
) -> int:
    """Records detection event with owner user_id."""
    conn = get_db_connection()
    cursor = conn.cursor()
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    target_user_id = user_id if user_id is not None else 1
    
    cursor.execute("""
        INSERT INTO events (datetime_str, camera_name, track_id, object_class, confidence, screenshot_filename, screenshot_path, source_type, user_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (now_str, camera_name, track_id, object_class, round(confidence, 2), screenshot_filename, screenshot_path, source_type, target_user_id))
    
    event_id = cursor.lastrowid
    conn.commit()
    conn.close()

    # Invalidate Redis cache on new event
    redis_cache.delete_prefix("events_")
    redis_cache.delete_prefix("stats")
    
    return event_id

def get_events(
    limit: int = 50,
    offset: int = 0,
    camera_name: Optional[str] = None,
    object_class: Optional[str] = None,
    user_id: Optional[int] = None,
    user_role: Optional[str] = None
) -> List[Dict[str, Any]]:
    conn = get_db_connection()
    cursor = conn.cursor()
    
    query = "SELECT * FROM events"
    params = []
    conditions = []
    
    if user_role != "admin" and user_id is not None:
        conditions.append("user_id = ?")
        params.append(user_id)

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

def get_events_paginated(
    page: int = 1,
    limit: int = 12,
    camera_name: Optional[str] = None,
    object_class: Optional[str] = None,
    user_id: Optional[int] = None,
    user_role: Optional[str] = None
) -> Dict[str, Any]:
    cache_key = f"events_p{page}_l{limit}_c{camera_name or 'all'}_o{object_class or 'all'}_u{user_id or 'all'}_r{user_role or 'user'}"
    cached_val = redis_cache.get_json(cache_key)
    if cached_val:
        return cached_val

    conn = get_db_connection()
    cursor = conn.cursor()
    
    where_clause = ""
    params = []
    conditions = []
    
    if user_role != "admin" and user_id is not None:
        conditions.append("user_id = ?")
        params.append(user_id)

    if camera_name:
        conditions.append("camera_name = ?")
        params.append(camera_name)
    if object_class:
        conditions.append("object_class = ?")
        params.append(object_class)
        
    if conditions:
        where_clause = " WHERE " + " AND ".join(conditions)
        
    count_query = f"SELECT COUNT(*) FROM events{where_clause}"
    cursor.execute(count_query, params)
    total = cursor.fetchone()[0]
    
    page = max(1, page)
    limit = max(1, limit)
    offset = (page - 1) * limit
    total_pages = max(1, math.ceil(total / limit))
    
    query = f"SELECT * FROM events{where_clause} ORDER BY id DESC LIMIT ? OFFSET ?"
    query_params = list(params)
    query_params.extend([limit, offset])
    
    cursor.execute(query, query_params)
    rows = cursor.fetchall()
    events = [dict(row) for row in rows]
    conn.close()
    
    result = {
        "events": events,
        "total": total,
        "page": page,
        "limit": limit,
        "total_pages": total_pages
    }
    
    redis_cache.set_json(cache_key, result, ttl_sec=15)
    return result

def delete_event(event_id: int) -> bool:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM events WHERE id = ?", (event_id,))
    affected = cursor.rowcount > 0
    conn.commit()
    conn.close()
    
    if affected:
        redis_cache.delete_prefix("events_")
        redis_cache.delete_prefix("stats")
    return affected

def clear_all_events() -> bool:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM events")
    conn.commit()
    conn.close()
    
    redis_cache.delete_prefix("events_")
    redis_cache.delete_prefix("stats")
    return True

def get_stats(user_id: Optional[int] = None, user_role: Optional[str] = None) -> Dict[str, Any]:
    cache_key = f"stats_u{user_id or 'all'}_r{user_role or 'user'}"
    cached_stats = redis_cache.get_json(cache_key)
    if cached_stats:
        return cached_stats

    conn = get_db_connection()
    cursor = conn.cursor()
    
    where_clause = ""
    params = []
    if user_role != "admin" and user_id is not None:
        where_clause = " WHERE user_id = ?"
        params.append(user_id)

    cursor.execute(f"SELECT COUNT(*) FROM events{where_clause}", params)
    total_events = cursor.fetchone()[0]
    
    today_str = datetime.now().strftime("%Y-%m-%d")
    today_where = f" WHERE datetime_str LIKE ?" if not where_clause else f"{where_clause} AND datetime_str LIKE ?"
    today_params = list(params) + [f"{today_str}%"]
    cursor.execute(f"SELECT COUNT(*) FROM events{today_where}", today_params)
    today_events = cursor.fetchone()[0]
    
    cursor.execute(f"SELECT datetime_str FROM events{where_clause} ORDER BY id DESC LIMIT 1", params)
    last_row = cursor.fetchone()
    last_detection = last_row[0] if last_row else "None"
    
    person_where = f" WHERE (object_class LIKE '%person%' OR object_class = 'person')" if not where_clause else f"{where_clause} AND (object_class LIKE '%person%' OR object_class = 'person')"
    cursor.execute(f"SELECT COUNT(*) FROM events{person_where}", params)
    persons_count = cursor.fetchone()[0]

    vehicle_where = f" WHERE (object_class LIKE '%car%' OR object_class LIKE '%vehicle%' OR object_class LIKE '%truck%' OR object_class LIKE '%bag%')" if not where_clause else f"{where_clause} AND (object_class LIKE '%car%' OR object_class LIKE '%vehicle%' OR object_class LIKE '%truck%' OR object_class LIKE '%bag%')"
    cursor.execute(f"SELECT COUNT(*) FROM events{vehicle_where}", params)
    vehicles_count = cursor.fetchone()[0]

    cursor.execute(f"SELECT AVG(confidence) FROM events{where_clause}", params)
    avg_row = cursor.fetchone()
    avg_confidence = round(float(avg_row[0]) * 100, 1) if avg_row and avg_row[0] is not None else 92.5
    
    conn.close()
    result = {
        "total_events": total_events,
        "today_events": today_events,
        "persons_count": persons_count,
        "vehicles_count": vehicles_count,
        "avg_confidence": avg_confidence,
        "last_detection": last_detection
    }
    redis_cache.set_json(cache_key, result, ttl_sec=15)
    return result

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

