from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import Response

from app.services.image_processing.filters import (
    InvalidImageError,
    gaussian_blur_from_bytes,
    grayscale_from_bytes,
)

router = APIRouter(
    tags=["Image Processing"],
)

_ALLOWED_TYPES = {"image/png", "image/jpeg", "image/jpg"}


def _validate_image_upload(file: UploadFile) -> None:
    if file.content_type not in _ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Use PNG or JPEG/JPG.",
        )


@router.post("/image-processing/grayscale")
async def grayscale(file: UploadFile = File(...)):
    _validate_image_upload(file)

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty file uploaded")

    try:
        output = grayscale_from_bytes(image_bytes)
    except InvalidImageError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return Response(content=output, media_type="image/png")


@router.post("/image-processing/blur/gaussian")
async def gaussian_blur(
    file: UploadFile = File(...),
    kernel_size: int = Query(..., gt=1),
    sigma: float = Query(..., ge=0.0),
):
    _validate_image_upload(file)

    if kernel_size % 2 == 0:
        raise HTTPException(
            status_code=400,
            detail="kernel_size must be an odd integer greater than 1",
        )

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty file uploaded")

    try:
        output = gaussian_blur_from_bytes(image_bytes, kernel_size=kernel_size, sigma=sigma)
    except InvalidImageError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return Response(content=output, media_type="image/png")
