import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


_ENDPOINTS = [
    "/api/v1/image-processing/grayscale",
    "/api/v1/image-processing/blur/gaussian?kernel_size=3&sigma=1.2",
    "/api/v1/image-processing/edges/sobel?kernel_size=3",
    "/api/v1/image-processing/edges/canny?threshold1=50&threshold2=150",
]


def _build_test_png(width: int, height: int) -> bytes:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:, :] = [0, 0, 255]
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    return encoded.tobytes()


def test_grayscale_success_returns_png_image():
    payload = _build_test_png(width=60, height=60)

    response = client.post(
        "/api/v1/image-processing/grayscale",
        files={"file": ("sample.png", payload, "image/png")},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("image/png")

    output = np.frombuffer(response.content, dtype=np.uint8)
    decoded = cv2.imdecode(output, cv2.IMREAD_UNCHANGED)
    assert decoded is not None
    assert len(decoded.shape) == 2


@pytest.mark.parametrize("endpoint", _ENDPOINTS)
def test_all_endpoints_reject_unsupported_content_type(endpoint: str):
    response = client.post(
        endpoint,
        files={"file": ("sample.txt", b"not-an-image", "text/plain")},
    )

    assert response.status_code == 400
    assert "Unsupported file type" in response.json()["detail"]


def test_gaussian_blur_success_returns_png_image():
    payload = _build_test_png(width=60, height=60)

    response = client.post(
        "/api/v1/image-processing/blur/gaussian?kernel_size=3&sigma=1.2",
        files={"file": ("sample.png", payload, "image/png")},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("image/png")

    output = np.frombuffer(response.content, dtype=np.uint8)
    decoded = cv2.imdecode(output, cv2.IMREAD_UNCHANGED)
    assert decoded is not None
    assert len(decoded.shape) == 3


def test_gaussian_blur_rejects_even_kernel_size():
    payload = _build_test_png(width=60, height=60)

    response = client.post(
        "/api/v1/image-processing/blur/gaussian?kernel_size=4&sigma=1.0",
        files={"file": ("sample.png", payload, "image/png")},
    )

    assert response.status_code == 400
    assert "kernel_size must be an odd integer greater than 1" in response.json()["detail"]


def test_sobel_success_returns_png_image():
    payload = _build_test_png(width=60, height=60)

    response = client.post(
        "/api/v1/image-processing/edges/sobel?kernel_size=3",
        files={"file": ("sample.png", payload, "image/png")},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("image/png")

    output = np.frombuffer(response.content, dtype=np.uint8)
    decoded = cv2.imdecode(output, cv2.IMREAD_UNCHANGED)
    assert decoded is not None
    assert len(decoded.shape) == 2


def test_sobel_rejects_even_kernel_size():
    payload = _build_test_png(width=60, height=60)

    response = client.post(
        "/api/v1/image-processing/edges/sobel?kernel_size=2",
        files={"file": ("sample.png", payload, "image/png")},
    )

    assert response.status_code == 400
    assert "kernel_size must be an odd integer" in response.json()["detail"]


def test_sobel_rejects_kernel_size_outside_allowed_set():
    payload = _build_test_png(width=60, height=60)

    response = client.post(
        "/api/v1/image-processing/edges/sobel?kernel_size=9",
        files={"file": ("sample.png", payload, "image/png")},
    )

    assert response.status_code == 400
    assert "kernel_size must be one of: 1, 3, 5, 7" in response.json()["detail"]


def test_canny_success_returns_png_image():
    payload = _build_test_png(width=60, height=60)

    response = client.post(
        "/api/v1/image-processing/edges/canny?threshold1=50&threshold2=150",
        files={"file": ("sample.png", payload, "image/png")},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("image/png")

    output = np.frombuffer(response.content, dtype=np.uint8)
    decoded = cv2.imdecode(output, cv2.IMREAD_UNCHANGED)
    assert decoded is not None
    assert len(decoded.shape) == 2


def test_canny_rejects_invalid_threshold_order():
    payload = _build_test_png(width=60, height=60)

    response = client.post(
        "/api/v1/image-processing/edges/canny?threshold1=170&threshold2=100",
        files={"file": ("sample.png", payload, "image/png")},
    )

    assert response.status_code == 400
    assert "threshold1 must be lower than threshold2" in response.json()["detail"]


@pytest.mark.parametrize("endpoint", _ENDPOINTS)
def test_all_endpoints_reject_empty_file(endpoint: str):
    response = client.post(
        endpoint,
        files={"file": ("sample.png", b"", "image/png")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Empty file uploaded"


@pytest.mark.parametrize("endpoint", _ENDPOINTS)
def test_all_endpoints_reject_invalid_image_bytes_with_image_content_type(endpoint: str):
    response = client.post(
        endpoint,
        files={"file": ("sample.png", b"not-an-image", "image/png")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid or unsupported image file"


@pytest.mark.parametrize("endpoint", _ENDPOINTS)
def test_all_endpoints_reject_image_smaller_than_min_50x50(endpoint: str):
    payload = _build_test_png(width=49, height=50)
    response = client.post(
        endpoint,
        files={"file": ("sample.png", payload, "image/png")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Image dimensions must be between 50x50 and 200x200 pixels"


@pytest.mark.parametrize("endpoint", _ENDPOINTS)
def test_all_endpoints_reject_image_larger_than_max_200x200(endpoint: str):
    payload = _build_test_png(width=201, height=200)
    response = client.post(
        endpoint,
        files={"file": ("sample.png", payload, "image/png")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Image dimensions must be between 50x50 and 200x200 pixels"


@pytest.mark.parametrize("endpoint", _ENDPOINTS)
def test_all_endpoints_accept_image_exactly_50x50(endpoint: str):
    payload = _build_test_png(width=50, height=50)
    response = client.post(
        endpoint,
        files={"file": ("sample.png", payload, "image/png")},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("image/png")


@pytest.mark.parametrize("endpoint", _ENDPOINTS)
def test_all_endpoints_accept_image_exactly_200x200(endpoint: str):
    payload = _build_test_png(width=200, height=200)
    response = client.post(
        endpoint,
        files={"file": ("sample.png", payload, "image/png")},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("image/png")
