class IAESError(Exception):
    """Base class for IAES exceptions."""


class ExtractionError(IAESError):
    """Raised when there's an error extracting forms or cells."""


class FormError(IAESError):
    """Base class for any exception that needs to include the top form in it."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args)
        self.top_form = kwargs.get('top_form')


class ValidationError(FormError):
    """Class for validation errors. This is needed to differentiate builtin exceptions with validation errors."""


class ScrollError(FormError):
    """Raised when user needs to scroll."""
