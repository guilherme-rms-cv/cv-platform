from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import Response

from app.services.image_processing.constraints import ALLOWED_CONTENT_TYPES
from app.services.image_processing.errors import (
    EmptyFileError,
    InvalidImageError,
    InvalidImageSizeError,
    UnsupportedContentTypeError,
)
from app.services.image_processing.filters import (
    apply_filter_from_bytes,
    canny_edges_from_bytes,
    gaussian_blur_from_bytes,
    grayscale_from_bytes,
    sobel_edges_from_bytes,
)
from app.services.image_processing.channels import (
    convert_color_space_from_bytes,
    extract_channel_from_bytes,
)
from app.services.image_processing.fundamentals import (
    image_metadata_from_bytes,
    kernel_window_from_bytes,
)
from app.services.image_processing.thresholding import (
    apply_threshold_from_bytes,
    is_binary_output_method,
)

router = APIRouter(
    tags=["Image Processing"],
)

def _validate_content_type(file: UploadFile) -> None:
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise UnsupportedContentTypeError()


async def _read_upload_image_bytes(file: UploadFile) -> bytes:
    _validate_content_type(file)
    image_bytes = await file.read()
    if not image_bytes:
        raise EmptyFileError()
    return image_bytes


def _as_bad_request(exc: Exception) -> HTTPException:
    return HTTPException(status_code=400, detail=str(exc))


@router.post("/image-processing/grayscale")
async def grayscale(file: UploadFile = File(...)):
    try:
        image_bytes = await _read_upload_image_bytes(file)
        output = grayscale_from_bytes(image_bytes)
    except (UnsupportedContentTypeError, EmptyFileError, InvalidImageError, InvalidImageSizeError) as exc:
        raise _as_bad_request(exc) from exc

    return Response(content=output, media_type="image/png")


@router.post("/image-processing/blur/gaussian")
async def gaussian_blur(
    file: UploadFile = File(...),
    kernel_size: int = Query(..., gt=1),
    sigma: float = Query(..., ge=0.0),
):
    if kernel_size % 2 == 0:
        raise HTTPException(
            status_code=400,
            detail="kernel_size must be an odd integer greater than 1",
        )

    try:
        image_bytes = await _read_upload_image_bytes(file)
        output = gaussian_blur_from_bytes(image_bytes, kernel_size=kernel_size, sigma=sigma)
    except (UnsupportedContentTypeError, EmptyFileError, InvalidImageError, InvalidImageSizeError) as exc:
        raise _as_bad_request(exc) from exc

    return Response(content=output, media_type="image/png")


@router.post("/image-processing/edges/sobel")
async def sobel_edges(
    file: UploadFile = File(...),
    kernel_size: int = Query(3, ge=1),
):
    if kernel_size % 2 == 0:
        raise HTTPException(
            status_code=400,
            detail="kernel_size must be an odd integer",
        )

    if kernel_size not in {1, 3, 5, 7}:
        raise HTTPException(
            status_code=400,
            detail="kernel_size must be one of: 1, 3, 5, 7",
        )

    try:
        image_bytes = await _read_upload_image_bytes(file)
        output = sobel_edges_from_bytes(image_bytes, kernel_size=kernel_size)
    except (UnsupportedContentTypeError, EmptyFileError, InvalidImageError, InvalidImageSizeError) as exc:
        raise _as_bad_request(exc) from exc

    return Response(content=output, media_type="image/png")


@router.post("/image-processing/edges/canny")
async def canny_edges(
    file: UploadFile = File(...),
    threshold1: float = Query(..., ge=0.0, le=255.0),
    threshold2: float = Query(..., ge=0.0, le=255.0),
):
    if threshold1 >= threshold2:
        raise HTTPException(
            status_code=400,
            detail="threshold1 must be lower than threshold2",
        )

    try:
        image_bytes = await _read_upload_image_bytes(file)
        output = canny_edges_from_bytes(
            image_bytes,
            threshold1=threshold1,
            threshold2=threshold2,
        )
    except (UnsupportedContentTypeError, EmptyFileError, InvalidImageError, InvalidImageSizeError) as exc:
        raise _as_bad_request(exc) from exc

    return Response(content=output, media_type="image/png")


@router.post("/image-processing/filters/apply")
async def filters_apply(
    file: UploadFile = File(...),
    filter_id: str = Query(...),
    kernel_size: int = Query(3, ge=1),
    sigma: float = Query(1.2, ge=0.0),
    sigma2: float = Query(2.4, ge=0.0),
    amount: float = Query(1.0, ge=0.0),
    d: int = Query(5, ge=0),
    sigma_color: float = Query(75.0, ge=0.0),
    sigma_space: float = Query(75.0, ge=0.0),
):
    try:
        image_bytes = await _read_upload_image_bytes(file)
        output = apply_filter_from_bytes(
            image_bytes=image_bytes,
            filter_id=filter_id,
            kernel_size=kernel_size,
            sigma=sigma,
            sigma2=sigma2,
            amount=amount,
            d=d,
            sigma_color=sigma_color,
            sigma_space=sigma_space,
        )
    except (
        UnsupportedContentTypeError,
        EmptyFileError,
        InvalidImageError,
        InvalidImageSizeError,
        ValueError,
    ) as exc:
        raise _as_bad_request(exc) from exc

    return Response(content=output, media_type="image/png")


@router.post("/image-processing/channels/convert")
async def channels_convert(
    file: UploadFile = File(...),
    target: str = Query(...),
):
    try:
        image_bytes = await _read_upload_image_bytes(file)
        output = convert_color_space_from_bytes(image_bytes=image_bytes, target=target)
    except (
        UnsupportedContentTypeError,
        EmptyFileError,
        InvalidImageError,
        InvalidImageSizeError,
        ValueError,
    ) as exc:
        raise _as_bad_request(exc) from exc

    return Response(content=output, media_type="image/png")


@router.post("/image-processing/channels/extract")
async def channels_extract(
    file: UploadFile = File(...),
    channel_id: str = Query(...),
):
    try:
        image_bytes = await _read_upload_image_bytes(file)
        output = extract_channel_from_bytes(image_bytes=image_bytes, channel_id=channel_id)
    except (
        UnsupportedContentTypeError,
        EmptyFileError,
        InvalidImageError,
        InvalidImageSizeError,
        ValueError,
    ) as exc:
        raise _as_bad_request(exc) from exc

    return Response(content=output, media_type="image/png")


@router.post("/image-processing/fundamentals/metadata")
async def fundamentals_metadata(file: UploadFile = File(...)):
    try:
        image_bytes = await _read_upload_image_bytes(file)
        metadata = image_metadata_from_bytes(image_bytes)
    except (UnsupportedContentTypeError, EmptyFileError, InvalidImageError, InvalidImageSizeError) as exc:
        raise _as_bad_request(exc) from exc

    return metadata


@router.post("/image-processing/fundamentals/kernel-window")
async def fundamentals_kernel_window(
    file: UploadFile = File(...),
    kernel_size: int = Query(3, ge=3),
    x: int = Query(0, ge=0),
    y: int = Query(0, ge=0),
    mode: str = Query("kernel"),
):
    try:
        image_bytes = await _read_upload_image_bytes(file)
        payload = kernel_window_from_bytes(
            image_bytes=image_bytes,
            kernel_size=kernel_size,
            x=x,
            y=y,
            mode=mode,
        )
    except (
        UnsupportedContentTypeError,
        EmptyFileError,
        InvalidImageError,
        InvalidImageSizeError,
        ValueError,
    ) as exc:
        raise _as_bad_request(exc) from exc

    return payload


@router.post("/image-processing/thresholding/apply")
async def thresholding_apply(
    file: UploadFile = File(...),
    method: str = Query(...),
    threshold: float = Query(127.0, ge=0.0, le=255.0),
    max_value: float = Query(255.0, ge=0.0, le=255.0),
    block_size: int = Query(11, ge=3),
    c: float = Query(2.0),
):
    try:
        image_bytes = await _read_upload_image_bytes(file)
        output = apply_threshold_from_bytes(
            image_bytes=image_bytes,
            method=method,
            threshold=threshold,
            max_value=max_value,
            block_size=block_size,
            c=c,
        )
    except (
        UnsupportedContentTypeError,
        EmptyFileError,
        InvalidImageError,
        InvalidImageSizeError,
        ValueError,
    ) as exc:
        raise _as_bad_request(exc) from exc

    headers = {"X-Binary-Output": "true" if is_binary_output_method(method) else "false"}
    return Response(content=output, media_type="image/png", headers=headers)
