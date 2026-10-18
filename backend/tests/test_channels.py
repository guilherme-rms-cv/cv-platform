import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def _build_test_png(width: int, height: int) -> bytes:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:, :] = [10, 120, 240]
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    return encoded.tobytes()


@pytest.mark.parametrize(
    ("target", "expected_channel_count"),
    [
        ("gray", 1),
        ("hsv", 3),
        ("hsl", 3),
        ("hsi", 3),
        ("ycbcr", 3),
        ("cmyk", 4),
    ],
)
def test_channels_convert_supported_targets(target: str, expected_channel_count: int):
    payload = _build_test_png(width=80, height=80)
    response = client.post(
        f"/api/v1/image-processing/channels/convert?target={target}",
        files={"file": ("sample.png", payload, "image/png")},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("image/png")

    decoded = cv2.imdecode(np.frombuffer(response.content, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    assert decoded is not None
    if expected_channel_count == 1:
        assert len(decoded.shape) == 2
    else:
        assert len(decoded.shape) == 3
        assert decoded.shape[2] == expected_channel_count

    assert decoded.min() >= 0
    assert decoded.max() <= 255


def test_channels_convert_rejects_invalid_target():
    payload = _build_test_png(width=80, height=80)
    response = client.post(
        "/api/v1/image-processing/channels/convert?target=invalid",
        files={"file": ("sample.png", payload, "image/png")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "target must be one of: gray, hsv, hsl, hsi, ycbcr, cmyk"


@pytest.mark.parametrize(
    "channel_id",
    ["r", "g", "b", "h", "s", "v", "l", "a", "b_lab", "y", "cb", "cr"],
)
def test_channels_extract_supported_channel_ids(channel_id: str):
    payload = _build_test_png(width=90, height=90)
    response = client.post(
        f"/api/v1/image-processing/channels/extract?channel_id={channel_id}",
        files={"file": ("sample.png", payload, "image/png")},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("image/png")

    decoded = cv2.imdecode(np.frombuffer(response.content, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    assert decoded is not None
    assert len(decoded.shape) == 2
    assert decoded.shape[0] == 90
    assert decoded.shape[1] == 90
    assert decoded.min() >= 0
    assert decoded.max() <= 255


def test_channels_extract_rejects_invalid_channel_id():
    payload = _build_test_png(width=90, height=90)
    response = client.post(
        "/api/v1/image-processing/channels/extract?channel_id=invalid",
        files={"file": ("sample.png", payload, "image/png")},
    )

    assert response.status_code == 400
    assert (
        response.json()["detail"]
        == "channel_id must be one of: r, g, b, h, s, v, l, a, b_lab, y, cb, cr"
    )
