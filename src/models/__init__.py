__all__ = ["AttentionUNet", "UNet", "build_model", "get_model_name"]


def get_model_name(config: dict, checkpoint: dict | None = None) -> str:
    if checkpoint is not None:
        checkpoint_config = checkpoint.get("config") or {}
        model_name = checkpoint_config.get("model")
        if model_name is not None:
            return str(model_name)
    return str(config.get("model", "unet"))


def build_model(config: dict, checkpoint: dict | None = None, in_channels: int = 3, out_channels: int = 1):
    model_name = get_model_name(config, checkpoint)
    if model_name == "unet":
        from src.models.unet import UNet

        return UNet(in_channels=in_channels, out_channels=out_channels)
    if model_name == "attention_unet":
        from src.models.attention_unet import AttentionUNet

        return AttentionUNet(in_channels=in_channels, out_channels=out_channels)
    raise ValueError(f"Unsupported model: {model_name!r}. Expected 'unet' or 'attention_unet'.")


def __getattr__(name: str):
    if name == "UNet":
        from src.models.unet import UNet

        return UNet
    if name == "AttentionUNet":
        from src.models.attention_unet import AttentionUNet

        return AttentionUNet
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
