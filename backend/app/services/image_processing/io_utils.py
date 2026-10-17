import cv2
import numpy as np

from app.services.image_processing.constraints import MAX_IMAGE_SIZE, MIN_IMAGE_SIZE
from app.services.image_processing.errors import InvalidImageError, InvalidImageSizeError


def decode_image(image_bytes: bytes) -> np.ndarray:
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if image is None:
        raise InvalidImageError()
    return image


def encode_png(image: np.ndarray) -> bytes:
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError("Failed to encode image as PNG")
    return encoded.tobytes()


def validate_dimensions(
    image: np.ndarray,
    min_size: int = MIN_IMAGE_SIZE,
    max_size: int = MAX_IMAGE_SIZE,
) -> None:
    height, width = image.shape[:2]
    if not (min_size <= height <= max_size and min_size <= width <= max_size):
        raise InvalidImageSizeError()


def count_pixels(image: np.ndarray) -> int:
    height, width = image.shape[:2]
    return height * width
