from app.services.image_processing.errors import (
    EmptyFileError,
    InvalidImageError,
    InvalidImageSizeError,
    UnsupportedContentTypeError,
)


def test_domain_error_default_messages():
    assert str(UnsupportedContentTypeError()) == "Unsupported file type. Use PNG or JPEG/JPG."
    assert str(EmptyFileError()) == "Empty file uploaded"
    assert str(InvalidImageError()) == "Invalid or unsupported image file"
    assert str(InvalidImageSizeError()) == "Image dimensions must be between 50x50 and 200x200 pixels"
