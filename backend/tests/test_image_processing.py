import cv2
import numpy as np
from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def _build_test_png() -> bytes:
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    image[:, :] = [0, 0, 255]
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    return encoded.tobytes()


def test_grayscale_success_returns_png_image():
    payload = _build_test_png()

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


def test_grayscale_rejects_unsupported_content_type():
    response = client.post(
        "/api/v1/image-processing/grayscale",
        files={"file": ("sample.txt", b"not-an-image", "text/plain")},
    )

    assert response.status_code == 400
    assert "Unsupported file type" in response.json()["detail"]


def test_gaussian_blur_success_returns_png_image():
    payload = _build_test_png()

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
    payload = _build_test_png()

    response = client.post(
        "/api/v1/image-processing/blur/gaussian?kernel_size=4&sigma=1.0",
        files={"file": ("sample.png", payload, "image/png")},
    )

    assert response.status_code == 400
    assert "kernel_size must be an odd integer greater than 1" in response.json()["detail"]


def test_gaussian_blur_rejects_unsupported_content_type():
    response = client.post(
        "/api/v1/image-processing/blur/gaussian?kernel_size=3&sigma=0.5",
        files={"file": ("sample.txt", b"not-an-image", "text/plain")},
    )

    assert response.status_code == 400
    assert "Unsupported file type" in response.json()["detail"]


def test_sobel_success_returns_png_image():
    payload = _build_test_png()

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
    payload = _build_test_png()

    response = client.post(
        "/api/v1/image-processing/edges/sobel?kernel_size=2",
        files={"file": ("sample.png", payload, "image/png")},
    )

    assert response.status_code == 400
    assert "kernel_size must be an odd integer" in response.json()["detail"]


def test_sobel_rejects_unsupported_content_type():
    response = client.post(
        "/api/v1/image-processing/edges/sobel?kernel_size=3",
        files={"file": ("sample.txt", b"not-an-image", "text/plain")},
    )

    assert response.status_code == 400
    assert "Unsupported file type" in response.json()["detail"]
