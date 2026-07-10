"""FastAPI application factory and entry point.

Run in development with:
    uvicorn app.main:app --reload --app-dir backend
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app import __version__
from app.api import meta, patients
from app.config import REPO_ROOT, get_settings

FRONTEND_DIST = REPO_ROOT / "frontend" / "dist"


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title=f"{settings.app_name} API",
        version=__version__,
        summary="Research/educational limb-scanning API — not a medical device.",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # API routes are namespaced under /api so the SPA can own the root path.
    app.include_router(meta.router, prefix="/api")
    app.include_router(patients.router, prefix="/api")

    _mount_frontend(app)
    return app


def _mount_frontend(app: FastAPI) -> None:
    """Serve the built SPA if present; otherwise return a clear message at root.

    We deliberately do not fake a frontend — in development the Vite dev server runs
    separately and talks to this API over CORS.
    """
    if FRONTEND_DIST.is_dir() and (FRONTEND_DIST / "index.html").exists():
        app.mount("/", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="spa")
    else:

        @app.get("/", include_in_schema=False)
        def _no_frontend() -> JSONResponse:
            return JSONResponse(
                {
                    "message": (
                        "MelaNone API is running. Built frontend not found at "
                        f"{Path('frontend/dist').as_posix()} — run the Vite dev server "
                        "(npm run dev) or build the frontend."
                    ),
                    "docs": "/docs",
                    "health": "/api/health",
                }
            )


app = create_app()
