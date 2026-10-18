import cv2
import numpy as np

from app.services.image_processing.io_utils import decode_image, encode_png, validate_dimensions


def _to_ycbcr(image_bgr: np.ndarray) -> np.ndarray:
    ycrcb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)
    y = ycrcb[:, :, 0]
    cr = ycrcb[:, :, 1]
    cb = ycrcb[:, :, 2]
    return np.dstack((y, cb, cr))


def _to_hsi(image_bgr: np.ndarray) -> np.ndarray:
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    r = image_rgb[:, :, 0]
    g = image_rgb[:, :, 1]
    b = image_rgb[:, :, 2]

    intensity = (r + g + b) / 3.0

    min_rgb = np.minimum(np.minimum(r, g), b)
    saturation = np.zeros_like(intensity)
    non_zero_intensity = intensity > 1e-6
    saturation[non_zero_intensity] = 1.0 - (min_rgb[non_zero_intensity] / intensity[non_zero_intensity])

    numerator = 0.5 * ((r - g) + (r - b))
    denominator = np.sqrt((r - g) ** 2 + (r - b) * (g - b)) + 1e-8
    theta = np.arccos(np.clip(numerator / denominator, -1.0, 1.0))
    hue = np.where(b <= g, theta, (2 * np.pi) - theta)

    hue_u8 = np.clip((hue / (2 * np.pi)) * 255.0, 0, 255).astype(np.uint8)
    saturation_u8 = np.clip(saturation * 255.0, 0, 255).astype(np.uint8)
    intensity_u8 = np.clip(intensity * 255.0, 0, 255).astype(np.uint8)
    return np.dstack((hue_u8, saturation_u8, intensity_u8))


def _to_cmyk(image_bgr: np.ndarray) -> np.ndarray:
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    r = image_rgb[:, :, 0]
    g = image_rgb[:, :, 1]
    b = image_rgb[:, :, 2]

    k = 1.0 - np.maximum(np.maximum(r, g), b)
    denominator = 1.0 - k
    safe_denominator = np.where(denominator < 1e-6, 1.0, denominator)

    c = np.where(denominator < 1e-6, 0.0, (1.0 - r - k) / safe_denominator)
    m = np.where(denominator < 1e-6, 0.0, (1.0 - g - k) / safe_denominator)
    y = np.where(denominator < 1e-6, 0.0, (1.0 - b - k) / safe_denominator)

    c_u8 = np.clip(c * 255.0, 0, 255).astype(np.uint8)
    m_u8 = np.clip(m * 255.0, 0, 255).astype(np.uint8)
    y_u8 = np.clip(y * 255.0, 0, 255).astype(np.uint8)
    k_u8 = np.clip(k * 255.0, 0, 255).astype(np.uint8)
    return np.dstack((c_u8, m_u8, y_u8, k_u8))


def _convert(image_bgr: np.ndarray, target: str) -> np.ndarray:
    target_normalized = target.lower()
    if target_normalized in {"gray", "grayscale"}:
        return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    if target_normalized == "hsv":
        return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    if target_normalized == "hsl":
        return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HLS)
    if target_normalized == "hsi":
        return _to_hsi(image_bgr)
    if target_normalized in {"ycbcr", "ycrcb"}:
        return _to_ycbcr(image_bgr)
    if target_normalized == "cmyk":
        return _to_cmyk(image_bgr)
    raise ValueError("target must be one of: gray, hsv, hsl, hsi, ycbcr, cmyk")


def convert_color_space_from_bytes(image_bytes: bytes, target: str) -> bytes:
    image = decode_image(image_bytes)
    validate_dimensions(image)
    converted = _convert(image, target=target)
    return encode_png(converted)


def _channel_matrix(image_bgr: np.ndarray, channel_id: str) -> np.ndarray:
    normalized = channel_id.lower()

    if normalized == "r":
        return image_bgr[:, :, 2]
    if normalized == "g":
        return image_bgr[:, :, 1]
    if normalized == "b":
        return image_bgr[:, :, 0]

    if normalized in {"h", "s", "v"}:
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        index = {"h": 0, "s": 1, "v": 2}[normalized]
        return hsv[:, :, index]

    if normalized in {"l", "a", "b_lab", "lab_l", "lab_a", "lab_b"}:
        lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
        index = {
            "l": 0,
            "lab_l": 0,
            "a": 1,
            "lab_a": 1,
            "b_lab": 2,
            "lab_b": 2,
        }[normalized]
        return lab[:, :, index]

    if normalized in {"y", "cb", "cr"}:
        ycbcr = _to_ycbcr(image_bgr)
        index = {"y": 0, "cb": 1, "cr": 2}[normalized]
        return ycbcr[:, :, index]

    raise ValueError("channel_id must be one of: r, g, b, h, s, v, l, a, b_lab, y, cb, cr")


def extract_channel_from_bytes(image_bytes: bytes, channel_id: str) -> bytes:
    image = decode_image(image_bytes)
    validate_dimensions(image)
    channel = _channel_matrix(image, channel_id=channel_id)
    return encode_png(channel)
