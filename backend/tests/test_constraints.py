from app.services.image_processing.constraints import (
    ALLOWED_CONTENT_TYPES,
    MAX_IMAGE_SIZE,
    MIN_IMAGE_SIZE,
)


def test_image_processing_constraints_constants():
    assert MIN_IMAGE_SIZE == 50
    assert MAX_IMAGE_SIZE == 200
    assert ALLOWED_CONTENT_TYPES == {"image/png", "image/jpeg", "image/jpg"}
