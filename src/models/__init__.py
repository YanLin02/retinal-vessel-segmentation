__all__ = ["UNet"]


def __getattr__(name: str):
    if name == "UNet":
        from src.models.unet import UNet

        return UNet
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
