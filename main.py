import logging
import os
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import chat, documents, progress, graphiti, pinned_qa, conversations, views

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

app = FastAPI(title="PersonaID Backend", version="0.1.0")

_raw_origins = os.environ.get("ALLOWED_ORIGINS", "http://localhost:3000")
allow_origins = [o.strip() for o in _raw_origins.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

app.include_router(chat.router)
app.include_router(documents.router)
app.include_router(progress.router)
app.include_router(graphiti.router)
app.include_router(pinned_qa.router)
app.include_router(conversations.router)
app.include_router(views.router)


@app.get("/health")
def health():
    return {"status": "ok"}
