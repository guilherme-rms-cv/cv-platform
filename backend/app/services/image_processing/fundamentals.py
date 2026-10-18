import base64

from app.services.image_processing.io_utils import (
    count_pixels,
    decode_image,
    encode_png,
    validate_dimensions,
)


def image_metadata_from_bytes(image_bytes: bytes) -> dict[str, int]:
    image = decode_image(image_bytes)
    validate_dimensions(image)

    height, width = image.shape[:2]
    channels = image.shape[2] if len(image.shape) == 3 else 1
    return {
        "width": width,
        "height": height,
        "channels": channels,
        "total_pixels": count_pixels(image),
    }


def _extract_kernel_window(image, kernel_size: int, x: int, y: int):
    height, width = image.shape[:2]
    if x + kernel_size > width or y + kernel_size > height:
        raise ValueError("Kernel window exceeds image boundaries for given x, y, and kernel_size")
    return image[y : y + kernel_size, x : x + kernel_size]


def kernel_window_from_bytes(image_bytes: bytes, kernel_size: int, x: int, y: int, mode: str) -> dict:
    image = decode_image(image_bytes)
    validate_dimensions(image)

    if kernel_size not in {3, 4, 5}:
        raise ValueError("kernel_size must be one of: 3, 4, 5")

    if mode not in {"kernel", "full"}:
        raise ValueError("mode must be one of: kernel, full")

    height, width = image.shape[:2]
    channels = image.shape[2] if len(image.shape) == 3 else 1

    if mode == "full":
        return {
            "mode": "full",
            "width": width,
            "height": height,
            "channels": channels,
            "total_pixels": count_pixels(image),
        }

    window = _extract_kernel_window(image, kernel_size=kernel_size, x=x, y=y)
    window_png_bytes = encode_png(window)
    window_image_base64 = base64.b64encode(window_png_bytes).decode("ascii")

    return {
        "mode": "kernel",
        "kernel_size": kernel_size,
        "x": x,
        "y": y,
        "width": width,
        "height": height,
        "channels": channels,
        "window": window.tolist(),
        "window_image_mime_type": "image/png",
        "window_image_base64": window_image_base64,
    }
