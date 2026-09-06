import cv2
import numpy as np


class InvalidImageError(ValueError):
    """Raised when input bytes do not represent a valid image."""


def grayscale_from_bytes(image_bytes: bytes) -> bytes:
    """Convert an encoded image (PNG/JPEG bytes) to grayscale PNG bytes."""
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if image is None:
        raise InvalidImageError("Invalid or unsupported image file")

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    ok, encoded = cv2.imencode(".png", gray_image)
    if not ok:
        raise RuntimeError("Failed to encode grayscale image")

    return encoded.tobytes()


def gaussian_blur_from_bytes(image_bytes: bytes, kernel_size: int, sigma: float) -> bytes:
    """Apply Gaussian blur to an encoded image and return PNG bytes."""
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if image is None:
        raise InvalidImageError("Invalid or unsupported image file")

    blurred_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigmaX=sigma)

    ok, encoded = cv2.imencode(".png", blurred_image)
    if not ok:
        raise RuntimeError("Failed to encode blurred image")

    return encoded.tobytes()


def sobel_edges_from_bytes(image_bytes: bytes, kernel_size: int) -> bytes:
    """Apply Sobel edge detection to an encoded image and return PNG bytes."""
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if image is None:
        raise InvalidImageError("Invalid or unsupported image file")

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # A light blur stabilizes gradients and yields clearer educational edge maps.
    smoothed = cv2.GaussianBlur(gray_image, (3, 3), sigmaX=0)
    grad_x = cv2.Sobel(smoothed, cv2.CV_32F, 1, 0, ksize=kernel_size)
    grad_y = cv2.Sobel(smoothed, cv2.CV_32F, 0, 1, ksize=kernel_size)
    magnitude = cv2.magnitude(grad_x, grad_y)

    normalized = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    normalized_u8 = normalized.astype(np.uint8)
    _, edges = cv2.threshold(normalized_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    ok, encoded = cv2.imencode(".png", edges)
    if not ok:
        raise RuntimeError("Failed to encode Sobel edge image")

    return encoded.tobytes()
