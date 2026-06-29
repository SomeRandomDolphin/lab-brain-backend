"""
main.py — Project root entry point.

Run:
    python main.py
    # or
    uvicorn main:app --reload
"""

import uvicorn
from app.main import app
from app.core.config import cfg

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=cfg.server.host,
        port=cfg.server.port,
        reload=False,
    )
