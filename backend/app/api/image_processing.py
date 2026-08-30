from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import Response

from app.services.image_processing.filters import InvalidImageError, grayscale_from_bytes

router = APIRouter(
    tags=["Image Processing"],
)

_ALLOWED_TYPES = {"image/png", "image/jpeg", "image/jpg"}


@router.post("/image-processing/grayscale")
async def grayscale(file: UploadFile = File(...)):
    if file.content_type not in _ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Use PNG or JPEG/JPG.",
        )

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty file uploaded")

    try:
        output = grayscale_from_bytes(image_bytes)
    except InvalidImageError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return Response(content=output, media_type="image/png")
