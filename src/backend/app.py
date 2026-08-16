"""Deprecated launcher for the unified FastAPI application.

Use ``python -m src.qa_api.app``. This wrapper remains temporarily so existing
commands start the same application instead of maintaining a second backend.
"""

from __future__ import annotations

import logging
import os


def main() -> None:
    import uvicorn

    host = os.getenv("RAG_API_HOST", "127.0.0.1")
    port = int(os.getenv("RAG_API_PORT", "8000"))
    logging.getLogger("rag-backend").warning(
        "src.backend.app is deprecated; starting src.qa_api.app on %s:%d", host, port
    )
    uvicorn.run("src.qa_api.app:app", host=host, port=port, workers=1)


if __name__ == "__main__":
    main()
