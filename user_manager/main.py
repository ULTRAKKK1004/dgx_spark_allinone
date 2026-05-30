import logging
import os
import aiohttp
import json
from typing import Optional, List, Dict, Any
from datetime import datetime
from fastapi import FastAPI, Depends, HTTPException, Request, Response, Form
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from sqlalchemy import Column, Integer, String, DateTime, JSON, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Database Setup ---
DB_URL = "sqlite:///./users.db"
TOKEN_FILE = "/home/yanus/unified_ai_service/api_tokens.json"

def verify_custom_token(token: str) -> bool:
    if not os.path.exists(TOKEN_FILE):
        return False
    try:
        with open(TOKEN_FILE, "r") as f:
            tokens = json.load(f)
            return token in tokens
    except Exception:
        return False

engine = create_engine(DB_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    name = Column(String)
    picture = Column(String)
    login_count = Column(Integer, default=0)
    last_login = Column(DateTime)
    level = Column(String, default="뉴비")
    created_at = Column(DateTime, default=datetime.utcnow)

class ActivityLog(Base):
    __tablename__ = "activity_logs"
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, index=True)
    action = Column(String) # "login", "access", "task"
    details = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow)

class LevelPermission(Base):
    __tablename__ = "level_permissions"
    level = Column(String, primary_key=True)
    permissions = Column(JSON) # {"vllm": True, "comfyui": False, ...}

class UserQuota(Base):
    __tablename__ = "user_quotas"
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, index=True)
    type = Column(String) # "credits", "image_gen", etc.
    remaining = Column(Integer, default=100)
    limit = Column(Integer, default=100)

Base.metadata.create_all(bind=engine)

# Lightweight migration: add new columns to pre-existing SQLite DBs.
with engine.begin() as _conn:
    from sqlalchemy import text as _text
    _existing_cols = {row[1] for row in _conn.execute(_text("PRAGMA table_info(users)")).fetchall()}
    if "picture" not in _existing_cols:
        _conn.execute(_text("ALTER TABLE users ADD COLUMN picture VARCHAR"))

# Initial Levels and default permissions
DEFAULT_LEVELS = [
    ("삭제된회원", {"vllm": False, "comfyui": False, "admin": False, "tube": False}),
    ("차단된회원", {"vllm": False, "comfyui": False, "admin": False, "tube": False}),
    ("뉴비", {"vllm": True, "comfyui": False, "admin": False, "tube": True}),
    ("일반회원", {"vllm": True, "comfyui": True, "admin": False, "tube": True}),
    ("우수회원", {"vllm": True, "comfyui": True, "admin": False, "tube": True}),
    ("준관리자", {"vllm": True, "comfyui": True, "admin": False, "tube": True}),
    ("관리자", {"vllm": True, "comfyui": True, "admin": True, "tube": True}),
]

_db = SessionLocal()
for lvl, perms in DEFAULT_LEVELS:
    if not _db.query(LevelPermission).filter(LevelPermission.level == lvl).first():
        _db.add(LevelPermission(level=lvl, permissions=perms))
_db.commit()
_db.close()

# --- FastAPI App ---
app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# --- Safe Migration: Ensure user_quotas is initialized for existing users ---
@app.on_event("startup")
async def startup_migration():
    db = SessionLocal()
    try:
        logger.info("Running safe DB migration checks...")
        # Add future migrations here
    finally:
        db.close()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- Middleware/Auth ---
ADMIN_EMAIL = "yeonwoo.kim03@gmail.com"

# --- Endpoints ---

async def _fetch_google_profile(access_token: str) -> dict:
    """Call Google's userinfo endpoint with the OAuth access token.

    Returns dict with 'name' and 'picture' on success, empty dict on failure.
    """
    if not access_token:
        return {}
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                "https://www.googleapis.com/oauth2/v3/userinfo",
                headers={"Authorization": f"Bearer {access_token}"},
                timeout=aiohttp.ClientTimeout(total=5),
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return {"name": data.get("name"), "picture": data.get("picture")}
                logger.warning(f"Google userinfo returned {resp.status}")
    except Exception as e:
        logger.warning(f"Google userinfo fetch failed: {e}")
    return {}


@app.get("/auth/verify")
async def verify_user(request: Request, db: Session = Depends(get_db)):
    host = request.headers.get("Host", "unknown")
    uri = request.headers.get("X-Original-URI", "unknown")
    cookie = request.headers.get("cookie", "")
    auth_header = request.headers.get("authorization", "")

    # 1. Strict Token-Only Auth for vllmapi.tor-ai.com
    if host == "vllmapi.tor-ai.com":
        if auth_header.startswith("Bearer "):
            token = auth_header.replace("Bearer ", "")
            if verify_custom_token(token):
                r = Response(status_code=200)
                r.headers["X-Auth-Request-Email"] = "api-user@tor-ai.com"
                r.headers["X-Auth-Request-User"] = "API User"
                return r
        logger.warning(f"Unauthorized API access attempt to {host}")
        return Response(status_code=401)

    # 2. General Auth (Custom Token OR OAuth2-Proxy)
    if auth_header.startswith("Bearer "):
        token = auth_header.replace("Bearer ", "")
        if verify_custom_token(token):
            r = Response(status_code=200)
            r.headers["X-Auth-Request-Email"] = "api-token-user@tor-ai.com"
            r.headers["X-Auth-Request-User"] = "API User"
            return r

    oauth2_url = "http://127.0.0.1:4180/oauth2/auth"
    logger.info(f"Verify Request - Host: {host}, URI: {uri}, HasCookie: {bool(cookie)}")

    forward_headers = {}
    if cookie:
        forward_headers["cookie"] = cookie
    if "authorization" in request.headers:
        forward_headers["authorization"] = request.headers["authorization"]

    # We must forward the host so oauth2-proxy knows which domain's cookie to look for
    forward_headers["Host"] = host

    async with aiohttp.ClientSession() as session:
        async with session.get(oauth2_url, headers=forward_headers) as resp:
            logger.info(f"OAuth2 Proxy Status: {resp.status} for {host}")
            if resp.status not in [200, 202]:
                return Response(status_code=resp.status)

            email = resp.headers.get("X-Auth-Request-Email") or resp.headers.get("x-auth-request-email")
            sub = resp.headers.get("X-Auth-Request-User") or resp.headers.get("x-auth-request-user")
            preferred = (
                resp.headers.get("X-Auth-Request-Preferred-Username")
                or resp.headers.get("x-auth-request-preferred-username")
            )
            access_token = (
                resp.headers.get("X-Auth-Request-Access-Token")
                or resp.headers.get("x-auth-request-access-token")
            )

    logger.info(f"Auth Success - Email: {email}, Sub: {sub}, Preferred: {preferred}, HasToken: {bool(access_token)}")

    if not email:
        logger.warning("Auth Success but no email in headers")
        return Response(status_code=401)

    user = db.query(User).filter(User.email == email).first()
    if not user:
        logger.info(f"Creating new user: {email}")
        user = User(email=email, name=preferred or sub)
        db.add(user)
        db.commit()
        db.refresh(user)

    # Sync profile info if we have a token
    if access_token:
        profile = await _fetch_google_profile(access_token)
        if profile:
            if profile.get("name") and not user.name:
                user.name = profile["name"]
            if profile.get("picture"):
                user.picture = profile["picture"]
            db.commit()

    log = ActivityLog(email=email, action="access", details=f"Access {host}{uri}")
    db.add(log)
    db.commit()

    # Ensure quota exists
    q = db.query(UserQuota).filter(UserQuota.email == email, UserQuota.type == "credits").first()
    if not q:
        db.add(UserQuota(email=email, type="credits", remaining=1000, limit=1000))
        db.commit()

    def _ok_response() -> Response:
        r = Response(status_code=200)
        r.headers["X-Auth-Request-User"] = user.name or sub or ""
        r.headers["X-Auth-Request-Email"] = email
        r.headers["X-Auth-Request-Preferred-Username"] = user.name or preferred or ""
        if user.picture:
            r.headers["X-Auth-Request-Image"] = user.picture
        return r

    # Admin check override
    if email == ADMIN_EMAIL:
        logger.info(f"Admin Bypass for {email}")
        return _ok_response()

    # Level check
    perms = db.query(LevelPermission).filter(LevelPermission.level == user.level).first()
    if not perms:
        logger.warning(f"No permissions found for level: {user.level}")
        return Response(status_code=403)

    return _ok_response()

@app.get("/admin", response_class=HTMLResponse)
async def admin_dashboard(request: Request, db: Session = Depends(get_db)):
    email = request.headers.get("X-Email")
    logger.info(f"Admin Dashboard Access - Email: {email}")
    if email != ADMIN_EMAIL:
        return templates.TemplateResponse(request, "error_403.html", {"request": request})
    
    users = db.query(User).all()
    logs = db.query(ActivityLog).order_by(ActivityLog.timestamp.desc()).limit(100).all()
    levels = db.query(LevelPermission).all()
    return templates.TemplateResponse(request, "admin.html", {"request": request, "users": users, "logs": logs, "levels": levels, "admin_email": ADMIN_EMAIL})

@app.get("/admin/test_report")
async def get_test_report(request: Request):
    email = request.headers.get("X-Email")
    if email != ADMIN_EMAIL:
        return JSONResponse(status_code=403, content={"error": "Forbidden"})
    
    report_path = "/home/yanus/test_report.json"
    if os.path.exists(report_path):
        with open(report_path, "r") as f:
            return JSONResponse(content=json.load(f))
    return JSONResponse(content=[])

@app.post("/admin/user/{user_id}/level")
async def update_user_level(user_id: int, level: str = Form(...), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == user_id).first()
    if user:
        user.level = level
        db.commit()
    return RedirectResponse(url="/admin", status_code=303)

@app.get("/error_403", response_class=HTMLResponse)
async def access_denied(request: Request):
    return templates.TemplateResponse(request, "error_403.html", {"request": request})

@app.get("/api/user/quota")
async def get_user_quota(email: str, db: Session = Depends(get_db)):
    quotas = db.query(UserQuota).filter(UserQuota.email == email).all()
    return {q.type: {"remaining": q.remaining, "limit": q.limit} for q in quotas}

@app.post("/api/user/quota/deduct")
async def deduct_user_quota(email: str, type: str, amount: int = 1, db: Session = Depends(get_db)):
    quota = db.query(UserQuota).filter(UserQuota.email == email, UserQuota.type == type).first()
    if not quota or quota.remaining < amount:
        return JSONResponse(status_code=403, content={"detail": "Insufficient quota"})
    
    quota.remaining -= amount
    db.commit()
    return {"remaining": quota.remaining}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
