import cv2
import numpy as np
from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def _build_test_png(width: int, height: int) -> bytes:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:, :] = [10, 20, 30]
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    return encoded.tobytes()


def test_fundamentals_metadata_success():
    payload = _build_test_png(width=60, height=70)
    response = client.post(
        "/api/v1/image-processing/fundamentals/metadata",
        files={"file": ("sample.png", payload, "image/png")},
    )

    assert response.status_code == 200
    assert response.json() == {
        "width": 60,
        "height": 70,
        "channels": 3,
        "total_pixels": 4200,
    }


def test_fundamentals_kernel_window_success():
    payload = _build_test_png(width=60, height=60)
    response = client.post(
        "/api/v1/image-processing/fundamentals/kernel-window?kernel_size=3&x=10&y=12&mode=kernel",
        files={"file": ("sample.png", payload, "image/png")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["mode"] == "kernel"
    assert body["kernel_size"] == 3
    assert body["x"] == 10
    assert body["y"] == 12
    assert body["width"] == 60
    assert body["height"] == 60
    assert body["channels"] == 3
    assert len(body["window"]) == 3
    assert len(body["window"][0]) == 3
    assert len(body["window"][0][0]) == 3


def test_fundamentals_kernel_window_full_mode_success():
    payload = _build_test_png(width=80, height=50)
    response = client.post(
        "/api/v1/image-processing/fundamentals/kernel-window?kernel_size=5&x=0&y=0&mode=full",
        files={"file": ("sample.png", payload, "image/png")},
    )

    assert response.status_code == 200
    assert response.json() == {
        "mode": "full",
        "width": 80,
        "height": 50,
        "channels": 3,
        "total_pixels": 4000,
    }


def test_fundamentals_kernel_window_rejects_invalid_kernel_size():
    payload = _build_test_png(width=60, height=60)
    response = client.post(
        "/api/v1/image-processing/fundamentals/kernel-window?kernel_size=6&x=0&y=0&mode=kernel",
        files={"file": ("sample.png", payload, "image/png")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "kernel_size must be one of: 3, 4, 5"


def test_fundamentals_kernel_window_rejects_invalid_mode():
    payload = _build_test_png(width=60, height=60)
    response = client.post(
        "/api/v1/image-processing/fundamentals/kernel-window?kernel_size=3&x=0&y=0&mode=invalid",
        files={"file": ("sample.png", payload, "image/png")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "mode must be one of: kernel, full"


def test_fundamentals_kernel_window_rejects_out_of_bounds():
    payload = _build_test_png(width=60, height=60)
    response = client.post(
        "/api/v1/image-processing/fundamentals/kernel-window?kernel_size=5&x=58&y=58&mode=kernel",
        files={"file": ("sample.png", payload, "image/png")},
    )

    assert response.status_code == 400
    assert (
        response.json()["detail"]
        == "Kernel window exceeds image boundaries for given x, y, and kernel_size"
    )


def test_fundamentals_metadata_rejects_out_of_range_dimensions():
    payload = _build_test_png(width=49, height=50)
    response = client.post(
        "/api/v1/image-processing/fundamentals/metadata",
        files={"file": ("sample.png", payload, "image/png")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Image dimensions must be between 50x50 and 200x200 pixels"
