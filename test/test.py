# %%
import torch.utils
import torch.utils.data
from src.dataset import BDD100K
from torchvision.transforms import PILToTensor
from torchvision.datasets.coco import CocoDetection
from torch import nn

# %%
dataset = BDD100K(root="./bdd100k/images/100k/val", annFile="./bdd100k/labels/det_20/det_val_coco.json", transform=PILToTensor())

# %%
from torch.utils.data import DataLoader
def collection(data):
    images = []
    targes = []
    for sample in data:
        images += [sample[0]]
        targes += [sample[1]]
    return images, targes
dataloader = DataLoader(dataset=dataset, batch_size=8, shuffle=True, collate_fn=collection)

# %%
train_features, train_targets = next(iter(dataloader))
# %%
from torchvision.models.detection import fasterrcnn_resnet50_fpn
model = fasterrcnn_resnet50_fpn(num_classes=11)
y_hat = model(train_features, train_targets)
print(y_hat)