from fastapi import FastAPI

from app.api.health import router as health_router


app = FastAPI(
    title="Interactive Learning Platform API",
    description=(
        "API for an interactive learning platform focused on "
        "Image Processing, Computer Vision and Computer Graphics."
    ),
    version="0.1.0",
)

app.include_router(
    health_router,
    prefix="/api/v1",
)