"""
main.py — Project root entry point.
"""

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