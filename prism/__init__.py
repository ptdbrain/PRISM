"""PRISM mixed-precision quantization pipeline."""

__all__ = ["PRISM", "__version__"]

__version__ = "0.1.0"


def __getattr__(name: str):
    if name == "PRISM":
        from prism.api import PRISM

        return PRISM
    raise AttributeError(f"module 'prism' has no attribute {name!r}")
