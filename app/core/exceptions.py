"""Domain-level errors. Mapped to HTTP responses in `app.main`."""


class AppError(Exception):
    status_code = 500

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class NotFoundError(AppError):
    status_code = 404


class ConflictError(AppError):
    status_code = 409


class ValidationError(AppError):
    status_code = 422


class ProviderNotConfiguredError(AppError):
    status_code = 400


class UnsupportedFileTypeError(AppError):
    status_code = 415


class UpstreamError(AppError):
    """An external dependency (vLLM, LLM API, vector DB) failed."""

    status_code = 502
