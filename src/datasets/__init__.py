__all__ = ["DriveDataset"]


def __getattr__(name: str):
    if name == "DriveDataset":
        from src.datasets.drive_dataset import DriveDataset

        return DriveDataset
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
