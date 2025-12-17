from pathlib import Path
from typing import List, Tuple, Optional, Callable, Any
from PIL import Image
from torch.utils.data import Dataset


class TestDataset(Dataset):
    """
    Custom Dataset for loading test images from a directory structure.
    """

    def __init__(self, path: str | Path, transform: Optional[Callable] = None):
        """
        Args:
            path (str | Path): Path to the dataset directory.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.path = Path(path)
        self.files: List[Path] = list(self.path.rglob("*.jpg"))
        self.transform = transform

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[Any, int, str]:
        img_path = self.files[idx]
        img_name = img_path.name

        with img_path.open("rb") as f:
            image_pil = Image.open(f)
            image = image_pil.convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, 0, img_name
