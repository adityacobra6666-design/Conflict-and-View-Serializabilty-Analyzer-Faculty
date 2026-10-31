import os
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.exceptions import HTTPException as StarletteHTTPException
from app.database import engine, Base
from app.routes import analyze, compare, examples, history, export, ai, health, upload

# Create database tables if not created
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Conflict & View Serializability Analyzer API",
    description="Deterministic DBMS Intelligence Platform Backend",
    version="1.0.0"
)

# Global safe exception handler to avoid leaking internal tracebacks in 500 responses
@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    if isinstance(exc, StarletteHTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail}
        )
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal server error occurred. Please try again later."}
    )

# Production CORS configuration
default_origins = [
    "http://localhost:5173",
    "http://localhost:5174",
    "http://localhost:3000",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:5174",
    "http://127.0.0.1:3000",
]

origins = list(default_origins)
frontend_url_env = os.getenv("FRONTEND_URL", "").strip()
allowed_origins_env = os.getenv("ALLOWED_ORIGINS", "").strip()

for env_var in (frontend_url_env, allowed_origins_env):
    if env_var:
        for u in env_var.split(","):
            cleaned = u.strip().rstrip("/")
            if cleaned and cleaned not in origins:
                origins.append(cleaned)

# Support Render frontend subdomains via regex (e.g. https://*.onrender.com)
render_origin_regex = os.getenv("ALLOWED_ORIGIN_REGEX", r"^https:\/\/.*\.onrender\.com$")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_origin_regex=render_origin_regex,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-Requested-With", "Accept", "Origin"],
)

# Register routes
app.include_router(health.router)
app.include_router(analyze.router)
app.include_router(compare.router)
app.include_router(examples.router)
app.include_router(history.router)
app.include_router(export.router)
app.include_router(ai.router)
app.include_router(upload.router)

@app.get("/")
def root():
    return {
        "app": "Conflict & View Serializability Analyzer Backend",
        "docs": "/docs",
        "status": "running"
    }
