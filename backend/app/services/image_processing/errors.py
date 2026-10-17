class InvalidImageError(ValueError):
    def __init__(self, message: str = "Invalid or unsupported image file"):
        super().__init__(message)


class UnsupportedContentTypeError(ValueError):
    def __init__(self, message: str = "Unsupported file type. Use PNG or JPEG/JPG."):
        super().__init__(message)


class EmptyFileError(ValueError):
    def __init__(self, message: str = "Empty file uploaded"):
        super().__init__(message)


class InvalidImageSizeError(ValueError):
    def __init__(self, message: str = "Image dimensions must be between 50x50 and 200x200 pixels"):
        super().__init__(message)
