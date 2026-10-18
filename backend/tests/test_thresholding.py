import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def _build_gradient_png(width: int, height: int) -> bytes:
    gradient = np.tile(np.linspace(0, 255, width, dtype=np.uint8), (height, 1))
    image = cv2.merge([gradient, gradient, gradient])
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    return encoded.tobytes()


def _decode_response_png(content: bytes) -> np.ndarray:
    decoded = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    assert decoded is not None
    return decoded


@pytest.mark.parametrize(
    "method",
    ["simple", "binary", "binary_inverse", "otsu", "triangle", "adaptive_mean", "adaptive_gaussian", "local"],
)
def test_thresholding_binary_output_methods_return_binary_values(method: str):
    payload = _build_gradient_png(width=120, height=80)
    response = client.post(
        f"/api/v1/image-processing/thresholding/apply?method={method}&threshold=120&max_value=255&block_size=11&c=2",
        files={"file": ("sample.png", payload, "image/png")},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("image/png")
    assert response.headers["X-Binary-Output"] == "true"

    image = _decode_response_png(response.content)
    unique_values = set(np.unique(image).tolist())
    assert unique_values.issubset({0, 255})


@pytest.mark.parametrize("method", ["trunc", "to_zero", "to_zero_inverse"])
def test_thresholding_non_binary_methods_return_png(method: str):
    payload = _build_gradient_png(width=120, height=80)
    response = client.post(
        f"/api/v1/image-processing/thresholding/apply?method={method}&threshold=120&max_value=255",
        files={"file": ("sample.png", payload, "image/png")},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("image/png")
    assert response.headers["X-Binary-Output"] == "false"
    image = _decode_response_png(response.content)
    assert image.shape == (80, 120)


def test_thresholding_rejects_invalid_method():
    payload = _build_gradient_png(width=120, height=80)
    response = client.post(
        "/api/v1/image-processing/thresholding/apply?method=invalid",
        files={"file": ("sample.png", payload, "image/png")},
    )

    assert response.status_code == 400
    assert "method must be one of:" in response.json()["detail"]


def test_thresholding_rejects_even_block_size_for_adaptive():
    payload = _build_gradient_png(width=120, height=80)
    response = client.post(
        "/api/v1/image-processing/thresholding/apply?method=adaptive_mean&block_size=10",
        files={"file": ("sample.png", payload, "image/png")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "block_size must be an odd integer greater than or equal to 3"
