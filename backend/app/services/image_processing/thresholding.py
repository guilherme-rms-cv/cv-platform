import cv2

from app.services.image_processing.io_utils import decode_image, encode_png, validate_dimensions


_SUPPORTED_METHODS = {
    "simple",
    "binary",
    "binary_inverse",
    "trunc",
    "to_zero",
    "to_zero_inverse",
    "otsu",
    "triangle",
    "adaptive_mean",
    "adaptive_gaussian",
    "local",
}

_BINARY_OUTPUT_METHODS = {
    "simple",
    "binary",
    "binary_inverse",
    "otsu",
    "triangle",
    "adaptive_mean",
    "adaptive_gaussian",
    "local",
}


def is_binary_output_method(method: str) -> bool:
    return method.lower() in _BINARY_OUTPUT_METHODS


def apply_threshold_from_bytes(
    image_bytes: bytes,
    method: str,
    threshold: float = 127.0,
    max_value: float = 255.0,
    block_size: int = 11,
    c: float = 2.0,
) -> bytes:
    image = decode_image(image_bytes)
    validate_dimensions(image)

    method_key = method.lower()
    if method_key not in _SUPPORTED_METHODS:
        raise ValueError(
            f"method must be one of: {', '.join(sorted(_SUPPORTED_METHODS))}"
        )

    if not (0 <= threshold <= 255):
        raise ValueError("threshold must be between 0 and 255")
    if not (0 <= max_value <= 255):
        raise ValueError("max_value must be between 0 and 255")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if method_key in {"adaptive_mean", "adaptive_gaussian", "local"}:
        if block_size < 3 or block_size % 2 == 0:
            raise ValueError("block_size must be an odd integer greater than or equal to 3")
        adaptive_method = (
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C
            if method_key == "adaptive_gaussian"
            else cv2.ADAPTIVE_THRESH_MEAN_C
        )
        output = cv2.adaptiveThreshold(
            gray,
            maxValue=max_value,
            adaptiveMethod=adaptive_method,
            thresholdType=cv2.THRESH_BINARY,
            blockSize=block_size,
            C=c,
        )
        return encode_png(output)

    threshold_type_map = {
        "simple": cv2.THRESH_BINARY,
        "binary": cv2.THRESH_BINARY,
        "binary_inverse": cv2.THRESH_BINARY_INV,
        "trunc": cv2.THRESH_TRUNC,
        "to_zero": cv2.THRESH_TOZERO,
        "to_zero_inverse": cv2.THRESH_TOZERO_INV,
        "otsu": cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        "triangle": cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE,
    }
    threshold_type = threshold_type_map[method_key]
    threshold_used = 0 if method_key in {"otsu", "triangle"} else threshold
    _, output = cv2.threshold(gray, threshold_used, max_value, threshold_type)
    return encode_png(output)
