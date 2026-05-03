"""
app/run.py
==========
Entry point cho Flask web demo.

Chạy:
    python app/run.py
    hoặc từ thư mục gốc:
    python -m app.run
"""

import os
import sys

# Thêm thư mục gốc vào path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import create_app

app = create_app(os.environ.get("FLASK_ENV", "development"))

if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════╗")
    print("║   🎭 Facial Emotion Recognition — Web Demo              ║")
    print("╠══════════════════════════════════════════════════════════╣")
    print("║   URL: http://127.0.0.1:5000                            ║")
    print("║   Press CTRL+C to stop                                  ║")
    print("╚══════════════════════════════════════════════════════════╝")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
