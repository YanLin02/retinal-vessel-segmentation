__all__ = [
    "CHASEDB1Dataset",
    "DriveDataset",
    "build_eval_dataset",
    "build_predict_dataset",
    "build_train_val_datasets",
]


def __getattr__(name: str):
    if name == "DriveDataset":
        from src.datasets.drive_dataset import DriveDataset

        return DriveDataset
    if name == "CHASEDB1Dataset":
        from src.datasets.chasedb1_dataset import CHASEDB1Dataset

        return CHASEDB1Dataset
    if name in {"build_train_val_datasets", "build_eval_dataset", "build_predict_dataset"}:
        from src.datasets import factory

        return getattr(factory, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
