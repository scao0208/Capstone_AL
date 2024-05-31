from pathlib import Path
from typing import Callable, Tuple
from torch.utils.data import Dataset
from torchvision.datasets.coco import CocoDetection
from typing import Any

import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import glob
import logging

class BDD100K(CocoDetection):
    def __init__(self, root: str | Path, annFile: str, transform: Callable[..., Any] | None = None, target_transform: Callable[..., Any] | None = None, transforms: Callable[..., Any] | None = None) -> None:
        super().__init__(root, annFile, transform, target_transform, transforms)
        self.img_path = glob.glob(f'{root}/*')
        
      

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        images, labels = super().__getitem__(index)
        output = {}
        output["boxes"] = torch.tensor([label["bbox"] for label in labels])
        # if len(output["boxes"]) != 0:
        try:
            output["boxes"][:, 2:4] = output["boxes"][:, 0:2] + output["boxes"][:, 2:4]
            output["labels"] = torch.tensor([label["category_id"] for label in labels])
            return images/255, output, self.img_path[index], index
        except: 
            return self.__getitem__(index + 1)


class SampleBatches(BDD100K):

    def __init__(self, root: str | Path, annFile: str, feasible_list: list[int],transform: Callable[..., Any] | None = None, target_transform: Callable[..., Any] | None = None, transforms: Callable[..., Any] | None = None) -> None:
        super().__init__(root, annFile, transform, target_transform, transforms)
        self.feasible_list = feasible_list

    def __len__(self):
        return len(self.feasible_list)

    def __getitem__(self, index: int) -> Tuple[Any]:
        index = self.feasible_list[index]
        return super().__getitem__(index)