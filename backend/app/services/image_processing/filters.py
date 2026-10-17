import cv2
import numpy as np
from app.services.image_processing.io_utils import decode_image, encode_png, validate_dimensions


def grayscale_from_bytes(image_bytes: bytes) -> bytes:
    """Convert an encoded image (PNG/JPEG bytes) to grayscale PNG bytes."""
    image = decode_image(image_bytes)
    validate_dimensions(image)

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return encode_png(gray_image)


def gaussian_blur_from_bytes(image_bytes: bytes, kernel_size: int, sigma: float) -> bytes:
    """Apply Gaussian blur to an encoded image and return PNG bytes."""
    image = decode_image(image_bytes)
    validate_dimensions(image)

    blurred_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigmaX=sigma)
    return encode_png(blurred_image)


def sobel_edges_from_bytes(image_bytes: bytes, kernel_size: int) -> bytes:
    """Apply Sobel edge detection to an encoded image and return PNG bytes."""
    image = decode_image(image_bytes)
    validate_dimensions(image)

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # A light blur stabilizes gradients and yields clearer educational edge maps.
    smoothed = cv2.GaussianBlur(gray_image, (3, 3), sigmaX=0)
    grad_x = cv2.Sobel(smoothed, cv2.CV_32F, 1, 0, ksize=kernel_size)
    grad_y = cv2.Sobel(smoothed, cv2.CV_32F, 0, 1, ksize=kernel_size)
    magnitude = cv2.magnitude(grad_x, grad_y)

    normalized = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    normalized_u8 = normalized.astype(np.uint8)
    _, edges = cv2.threshold(normalized_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return encode_png(edges)


def canny_edges_from_bytes(image_bytes: bytes, threshold1: float, threshold2: float) -> bytes:
    """Apply Canny edge detection to an encoded image and return PNG bytes."""
    image = decode_image(image_bytes)
    validate_dimensions(image)

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    smoothed = cv2.GaussianBlur(gray_image, (5, 5), sigmaX=1.2)
    edges = cv2.Canny(smoothed, threshold1=threshold1, threshold2=threshold2)
    return encode_png(edges)
