import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def _build_test_png(width: int, height: int) -> bytes:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:, :] = [30, 140, 220]
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    return encoded.tobytes()


@pytest.mark.parametrize(
    "query",
    [
        "filter_id=mean&kernel_size=3",
        "filter_id=weighted_mean&kernel_size=5&sigma=1.1",
        "filter_id=gaussian&kernel_size=3&sigma=1.2",
        "filter_id=laplacian&kernel_size=3",
        "filter_id=sobel&kernel_size=3",
        "filter_id=prewitt",
        "filter_id=scharr",
        "filter_id=roberts",
        "filter_id=median&kernel_size=3",
        "filter_id=max&kernel_size=3",
        "filter_id=min&kernel_size=3",
        "filter_id=midpoint&kernel_size=5",
        "filter_id=bilateral&d=5&sigma_color=60&sigma_space=60",
        "filter_id=unsharp&kernel_size=3&sigma=1.0&amount=1.2",
        "filter_id=high_pass&kernel_size=3&sigma=1.0",
        "filter_id=low_pass&kernel_size=5&sigma=1.4",
        "filter_id=dog&kernel_size=5&sigma=1.0&sigma2=2.2",
        "filter_id=log&kernel_size=3&sigma=1.0",
    ],
)
def test_filters_phase1_supported_filters_return_png(query: str):
    payload = _build_test_png(width=90, height=90)
    response = client.post(
        f"/api/v1/image-processing/filters/apply?{query}",
        files={"file": ("sample.png", payload, "image/png")},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("image/png")

    decoded = cv2.imdecode(np.frombuffer(response.content, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    assert decoded is not None
    assert decoded.shape[0] == 90
    assert decoded.shape[1] == 90
    assert decoded.min() >= 0
    assert decoded.max() <= 255


def test_filters_phase1_rejects_invalid_filter_id():
    payload = _build_test_png(width=90, height=90)
    response = client.post(
        "/api/v1/image-processing/filters/apply?filter_id=invalid",
        files={"file": ("sample.png", payload, "image/png")},
    )

    assert response.status_code == 400
    assert "filter_id must be one of:" in response.json()["detail"]


@pytest.mark.parametrize(
    ("query", "expected_detail"),
    [
        (
            "filter_id=gaussian&kernel_size=4",
            "kernel_size must be an odd integer greater than or equal to 3",
        ),
        (
            "filter_id=median&kernel_size=1",
            "kernel_size must be an odd integer greater than or equal to 3",
        ),
        (
            "filter_id=bilateral&d=0",
            "d must be greater than 0 for bilateral filter",
        ),
        (
            "filter_id=dog&kernel_size=3&sigma=1.5&sigma2=1.0",
            "sigma2 must be greater than sigma for dog filter",
        ),
    ],
)
def test_filters_phase1_rejects_invalid_parameters(query: str, expected_detail: str):
    payload = _build_test_png(width=90, height=90)
    response = client.post(
        f"/api/v1/image-processing/filters/apply?{query}",
        files={"file": ("sample.png", payload, "image/png")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == expected_detail
