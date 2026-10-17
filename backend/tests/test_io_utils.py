import cv2
import numpy as np
import pytest

from app.services.image_processing.errors import InvalidImageError, InvalidImageSizeError
from app.services.image_processing.io_utils import (
    count_pixels,
    decode_image,
    encode_png,
    validate_dimensions,
)


def _build_test_png(width: int, height: int) -> bytes:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:, :] = [0, 255, 0]
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    return encoded.tobytes()


def test_decode_image_success():
    payload = _build_test_png(width=60, height=80)
    image = decode_image(payload)
    assert image is not None
    assert image.shape[0] == 80
    assert image.shape[1] == 60


def test_decode_image_raises_for_invalid_bytes():
    with pytest.raises(InvalidImageError, match="Invalid or unsupported image file"):
        decode_image(b"invalid-image-bytes")


def test_encode_png_success():
    image = np.zeros((50, 50, 3), dtype=np.uint8)
    output = encode_png(image)
    decoded = cv2.imdecode(np.frombuffer(output, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    assert decoded is not None


def test_validate_dimensions_accepts_bounds():
    validate_dimensions(np.zeros((50, 50, 3), dtype=np.uint8))
    validate_dimensions(np.zeros((200, 200, 3), dtype=np.uint8))


def test_validate_dimensions_raises_for_out_of_bounds():
    with pytest.raises(
        InvalidImageSizeError,
        match="Image dimensions must be between 50x50 and 200x200 pixels",
    ):
        validate_dimensions(np.zeros((49, 50, 3), dtype=np.uint8))

    with pytest.raises(
        InvalidImageSizeError,
        match="Image dimensions must be between 50x50 and 200x200 pixels",
    ):
        validate_dimensions(np.zeros((201, 200, 3), dtype=np.uint8))


def test_count_pixels():
    image = np.zeros((75, 60, 3), dtype=np.uint8)
    assert count_pixels(image) == 4500
