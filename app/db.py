import sqlite3
import os
import json
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "data", "history.db")


def get_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            emotion TEXT NOT NULL,
            confidence REAL NOT NULL,
            probabilities TEXT,
            model_type TEXT DEFAULT 'keras',
            face_count INTEGER DEFAULT 1,
            source TEXT DEFAULT 'webcam'
        )
    """)
    conn.commit()
    return conn


def save_prediction(emotion, confidence, probabilities, model_type="keras", face_count=1, source="webcam"):
    conn = get_db()
    conn.execute(
        "INSERT INTO predictions (timestamp, emotion, confidence, probabilities, model_type, face_count, source) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (datetime.now().isoformat(), emotion, confidence,
         json.dumps(probabilities), model_type, face_count, source),
    )
    conn.commit()
    conn.close()


def get_history(limit=100):
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM predictions ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_stats():
    conn = get_db()
    total = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    rows = conn.execute(
        "SELECT emotion, COUNT(*) as cnt, ROUND(AVG(confidence), 4) as avg_conf "
        "FROM predictions GROUP BY emotion ORDER BY cnt DESC"
    ).fetchall()
    conn.close()
    return {"total": total, "emotions": [dict(r) for r in rows]}
