"""
main.py — Project root entry point.
"""

# Load .env as the very first action of the process, before importing
# anything from `app` — app.main already calls load_env() itself, but doing
# it here too makes the ordering explicit and guarantees it happens even if
# some future import path (tests, scripts, alembic CLI, etc.) reaches
# app.core.config before app.main does. load_env() is idempotent/cheap to
# call twice.
from app.core.env import load_env
load_env()

import uvicorn
from app.main import app
from app.core.config import cfg

if __name__ == "__main__":
    uvicorn.run(
        app,
        host=cfg.server.host,
        port=cfg.server.port,
        reload=False,
        log_config=None,   # don't let uvicorn's dictConfig override setup_logging()
    )