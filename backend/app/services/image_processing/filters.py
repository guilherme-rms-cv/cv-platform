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


_SUPPORTED_FILTERS = {
    "mean",
    "weighted_mean",
    "gaussian",
    "laplacian",
    "sobel",
    "prewitt",
    "scharr",
    "roberts",
    "median",
    "max",
    "min",
    "midpoint",
    "bilateral",
    "unsharp",
    "high_pass",
    "low_pass",
    "dog",
    "log",
}


def _require_odd_kernel_size(kernel_size: int) -> None:
    if kernel_size < 3 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be an odd integer greater than or equal to 3")


def _grayscale(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _prewitt(gray: np.ndarray) -> np.ndarray:
    kernel_x = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=np.float32)
    kernel_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=np.float32)
    grad_x = cv2.filter2D(gray, cv2.CV_32F, kernel_x)
    grad_y = cv2.filter2D(gray, cv2.CV_32F, kernel_y)
    magnitude = cv2.magnitude(grad_x, grad_y)
    return cv2.convertScaleAbs(magnitude)


def _roberts(gray: np.ndarray) -> np.ndarray:
    kernel_x = np.array([[1, 0], [0, -1]], dtype=np.float32)
    kernel_y = np.array([[0, 1], [-1, 0]], dtype=np.float32)
    grad_x = cv2.filter2D(gray, cv2.CV_32F, kernel_x)
    grad_y = cv2.filter2D(gray, cv2.CV_32F, kernel_y)
    magnitude = cv2.magnitude(grad_x, grad_y)
    return cv2.convertScaleAbs(magnitude)


def apply_filter_from_bytes(
    image_bytes: bytes,
    filter_id: str,
    kernel_size: int = 3,
    sigma: float = 1.2,
    sigma2: float = 2.4,
    amount: float = 1.0,
    d: int = 5,
    sigma_color: float = 75.0,
    sigma_space: float = 75.0,
) -> bytes:
    image = decode_image(image_bytes)
    validate_dimensions(image)

    filter_key = filter_id.lower()
    if filter_key not in _SUPPORTED_FILTERS:
        raise ValueError(f"filter_id must be one of: {', '.join(sorted(_SUPPORTED_FILTERS))}")

    if filter_key in {
        "mean",
        "weighted_mean",
        "gaussian",
        "laplacian",
        "sobel",
        "median",
        "max",
        "min",
        "midpoint",
        "unsharp",
        "high_pass",
        "low_pass",
        "dog",
        "log",
    }:
        _require_odd_kernel_size(kernel_size)

    if sigma < 0:
        raise ValueError("sigma must be greater than or equal to 0")
    if sigma2 < 0:
        raise ValueError("sigma2 must be greater than or equal to 0")
    if filter_key == "dog" and sigma2 <= sigma:
        raise ValueError("sigma2 must be greater than sigma for dog filter")
    if amount < 0:
        raise ValueError("amount must be greater than or equal to 0")
    if filter_key == "bilateral" and d <= 0:
        raise ValueError("d must be greater than 0 for bilateral filter")

    if filter_key == "mean":
        output = cv2.blur(image, (kernel_size, kernel_size))
    elif filter_key == "weighted_mean":
        vector = cv2.getGaussianKernel(kernel_size, sigma if sigma > 0 else 0)
        kernel = vector @ vector.T
        output = cv2.filter2D(image, -1, kernel)
    elif filter_key == "gaussian":
        output = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigmaX=sigma)
    elif filter_key == "laplacian":
        output = cv2.convertScaleAbs(cv2.Laplacian(_grayscale(image), cv2.CV_32F, ksize=kernel_size))
    elif filter_key == "sobel":
        gray = _grayscale(image)
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=kernel_size)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=kernel_size)
        output = cv2.convertScaleAbs(cv2.magnitude(grad_x, grad_y))
    elif filter_key == "prewitt":
        output = _prewitt(_grayscale(image))
    elif filter_key == "scharr":
        gray = _grayscale(image)
        grad_x = cv2.Scharr(gray, cv2.CV_32F, 1, 0)
        grad_y = cv2.Scharr(gray, cv2.CV_32F, 0, 1)
        output = cv2.convertScaleAbs(cv2.magnitude(grad_x, grad_y))
    elif filter_key == "roberts":
        output = _roberts(_grayscale(image))
    elif filter_key == "median":
        output = cv2.medianBlur(image, kernel_size)
    elif filter_key == "max":
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        output = cv2.dilate(image, kernel)
    elif filter_key == "min":
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        output = cv2.erode(image, kernel)
    elif filter_key == "midpoint":
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        max_img = cv2.dilate(image, kernel)
        min_img = cv2.erode(image, kernel)
        output = ((max_img.astype(np.float32) + min_img.astype(np.float32)) / 2.0).astype(np.uint8)
    elif filter_key == "bilateral":
        output = cv2.bilateralFilter(image, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)
    elif filter_key == "unsharp":
        blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigmaX=sigma)
        output = cv2.addWeighted(image, 1.0 + amount, blurred, -amount, 0)
    elif filter_key == "high_pass":
        low = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigmaX=sigma)
        output = cv2.convertScaleAbs(cv2.subtract(image, low))
    elif filter_key == "low_pass":
        output = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigmaX=sigma)
    elif filter_key == "dog":
        blur1 = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigmaX=sigma)
        blur2 = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigmaX=sigma2)
        output = cv2.convertScaleAbs(cv2.subtract(blur2, blur1))
    else:  # log
        gray = _grayscale(image)
        blurred = cv2.GaussianBlur(gray, (kernel_size, kernel_size), sigmaX=sigma)
        output = cv2.convertScaleAbs(cv2.Laplacian(blurred, cv2.CV_32F, ksize=kernel_size))

    return encode_png(output)
