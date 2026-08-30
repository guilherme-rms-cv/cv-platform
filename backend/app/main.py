from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.health import router as health_router
from app.api.image_processing import router as image_processing_router


app = FastAPI(
    title="Interactive Learning Platform API",
    description=(
        "API for an interactive learning platform focused on "
        "Image Processing, Computer Vision and Computer Graphics."
    ),
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500",
        "http://127.0.0.1:8000",
        "http://localhost:8000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(
    health_router,
    prefix="/api/v1",
)

app.include_router(
    image_processing_router,
    prefix="/api/v1",
)