from fastapi import FastAPI, Request, Depends, HTTPException, status, Form
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Boolean, ForeignKey, Text, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
import datetime
import json
import os
import logging
import aiohttp
from fastapi import Response

# --- Logging Setup ---
# Force logging to be visible in nohup.out/service.log
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("user_manager")

# --- Database Setup ---
DB_URL = "sqlite:///./users.db"
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
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

class ActivityLog(Base):
    __tablename__ = "activity_logs"
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, index=True)
    action = Column(String)
    details = Column(Text)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)

class LevelPermission(Base):
    __tablename__ = "level_permissions"
    level = Column(String, primary_key=True)
    permissions = Column(JSON) # {"vllm": True, "comfyui": False, ...}

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

db = SessionLocal()
for level, perms in DEFAULT_LEVELS:
    if not db.query(LevelPermission).filter(LevelPermission.level == level).first():
        db.add(LevelPermission(level=level, permissions=perms))
db.commit()
db.close()

# --- FastAPI App ---
app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

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
    oauth2_url = "http://127.0.0.1:4180/oauth2/auth"

    host = request.headers.get("Host", "unknown")
    uri = request.headers.get("X-Original-URI", "unknown")
    cookie = request.headers.get("cookie", "")

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

    resource = "unknown"
    if "vllm" in host: resource = "vllm"
    elif "comfyui" in host: resource = "comfyui"
    elif "tube" in host: resource = "tube"
    elif "blog" in host: resource = "blog"
    elif "admin" in host or uri.startswith("/admin"): resource = "admin"

    user = db.query(User).filter(User.email == email).first()

    # Prefer the human-readable name from preferred_username; fall back to sub.
    display_name = preferred or (user.name if user else None) or sub or email.split("@")[0]

    needs_picture_fetch = user is None or not user.picture
    if needs_picture_fetch and access_token:
        profile = await _fetch_google_profile(access_token)
        if profile.get("name"):
            display_name = profile["name"]
        picture_url = profile.get("picture")
    else:
        picture_url = user.picture if user else None

    if not user:
        user = User(
            email=email,
            name=display_name,
            picture=picture_url,
            level="뉴비",
            login_count=1,
            last_login=datetime.datetime.utcnow(),
        )
        db.add(user)
        log = ActivityLog(email=email, action="signup", details=f"Access {host}{uri}")
        db.add(log)
        db.commit()
        db.refresh(user)
    else:
        user.login_count += 1
        user.last_login = datetime.datetime.utcnow()
        if display_name and user.name != display_name:
            user.name = display_name
        if picture_url and user.picture != picture_url:
            user.picture = picture_url
        log = ActivityLog(email=email, action="access", details=f"Access {host}{uri}")
        db.add(log)
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

    if user.level == "차단된회원" or user.level == "삭제된회원":
        logger.warning(f"Access Denied - Level: {user.level} for {email}")
        return Response(status_code=403)

    level_perm = db.query(LevelPermission).filter(LevelPermission.level == user.level).first()
    if level_perm:
        perms = level_perm.permissions
        if resource in perms and not perms[resource]:
            logger.warning(f"Access Denied - No perm for {resource} for {email}")
            return Response(status_code=403)

    return _ok_response()

@app.get("/admin", response_class=HTMLResponse)
async def admin_dashboard(request: Request, db: Session = Depends(get_db)):
    email = request.headers.get("X-Email")
    logger.info(f"Admin Dashboard Access - Email: {email}")
    if email != ADMIN_EMAIL:
        return templates.TemplateResponse(request=request, name="error_403.html", context={}, status_code=403)
    
    users = db.query(User).all()
    levels = db.query(LevelPermission).all()
    return templates.TemplateResponse(request=request, name="admin.html", context={"users": users, "levels": levels, "admin_email": ADMIN_EMAIL})

@app.post("/admin/update_user")
async def update_user(
    request: Request,
    email: str = Form(...),
    level: str = Form(...),
    db: Session = Depends(get_db)
):
    admin_email = request.headers.get("X-Email")
    if admin_email != ADMIN_EMAIL:
        return JSONResponse(status_code=403, content={"detail": "Unauthorized"})
    
    user = db.query(User).filter(User.email == email).first()
    if user:
        old_level = user.level
        user.level = level
        log = ActivityLog(email=admin_email, action="update_user", details=f"Changed {email} level from {old_level} to {level}")
        db.add(log)
        db.commit()
    return RedirectResponse(url="/admin", status_code=303)

@app.post("/admin/update_permissions")
async def update_permissions(
    request: Request,
    level: str = Form(...),
    vllm: bool = Form(False),
    comfyui: bool = Form(False),
    admin: bool = Form(False),
    tube: bool = Form(False),
    db: Session = Depends(get_db)
):
    admin_email = request.headers.get("X-Email")
    if admin_email != ADMIN_EMAIL:
        return JSONResponse(status_code=403, content={"detail": "Unauthorized"})
    
    perm = db.query(LevelPermission).filter(LevelPermission.level == level).first()
    if perm:
        perm.permissions = {"vllm": vllm, "comfyui": comfyui, "admin": admin, "tube": tube}
        log = ActivityLog(email=admin_email, action="update_permissions", details=f"Updated permissions for {level}")
        db.add(log)
        db.commit()
    return RedirectResponse(url="/admin", status_code=303)

@app.get("/admin/logs/{email}", response_class=HTMLResponse)
async def user_logs(request: Request, email: str, db: Session = Depends(get_db)):
    admin_email = request.headers.get("X-Email")
    if admin_email != ADMIN_EMAIL:
        return templates.TemplateResponse(request=request, name="error_403.html", context={}, status_code=403)
    
    logs = db.query(ActivityLog).filter(ActivityLog.email == email).order_by(ActivityLog.timestamp.desc()).limit(100).all()
    return templates.TemplateResponse(request=request, name="logs.html", context={"email": email, "logs": logs})

@app.get("/error_403", response_class=HTMLResponse)
async def access_denied(request: Request):
    return templates.TemplateResponse(request=request, name="error_403.html", context={})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
