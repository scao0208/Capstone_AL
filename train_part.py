import torch
import torch.utils.data
import numpy as np
from src.dataset import BDD100K
from src.trainingModule import WarmupMultiStepLR
from torchvision.transforms import PILToTensor

from torchmetrics.detection.mean_ap import MeanAveragePrecision

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from typing import Any

from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn




def collection(data):
    images = []
    targes = []
    for sample in data:
        images += [sample[0]]
        targes += [sample[1]]
    return images, targes


# class LitAutoTrain(FasterRCNNLightning):
#     def __init__(self, model: nn.Module, lr: float = 0.0001, iou_threshold: float = 0.5):
#         super().__init__(model, lr, iou_threshold)

class LitAutoTrain(L.LightningModule):
    def __init__(self, model, lr: float = 0.01, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.model= model
        self.lr = lr
        # outputs
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.steps = 0
        self.steps_validation = 0
    
    def forward(self, x, y):
        return self.model(x, y)
    
    def training_step(self, batch, batch_idx):
        # Batch
        # x, y, x_name, y_name = batch  # tuple unpacking
        if len(batch) == 2:
            x, y = batch
        elif len(batch) == 4:
            x, y, _, _ = batch

        loss_dict = self.model(x, y)
        loss = sum(loss for loss in loss_dict.values())
        # self.log_dict(loss_dict)
        self.logger.log_metrics(loss_dict, step=self.steps)
        self.steps += 1
        return loss

    def validation_step(self, batch, batch_idx):
        # Batch
        # x, y, x_name, y_name = batch
        if len(batch) == 2:
            x, y = batch
        elif len(batch) == 4:
            x, y, _, _ = batch
        y_hat = self.model(x)

        # calculate uncertainty 
        pred_scores = y_hat[0]['scores']
        val_entropy =  sum(-pred_scores * torch.log(pred_scores))

        # calculate MAP
        map = MeanAveragePrecision(box_format="xywh", iou_type="bbox")
        map.update(y_hat, y)
        val_map = map.compute()
        # val_map.pop("classes")
        val_map50 = map.compute()["map_50"].item()

        # calculate recall
        # recall = map.compute()["mar_large"].item()

        # combine the 3 above indicators in the validation accuracy
        # val_acc = 0.4 * val_map50 + 0.4 * recall + 0.2 * np.exp(- 1e-2 * uncertainty) 
        
        # self.log('val_acc', val_acc)
        # self.log('val_entropy', val_entropy)


        self.log_dict(val_map)
        val_map['my_map_50'] = val_map['map_50']
        self.logger.log_metrics(val_map, step=self.steps_validation)
        self.steps_validation += 1
        print(f"steps: {self.steps_validation}")

        # return val_acc
        return val_map50
        

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.model.parameters(), lr=self.lr, momentum=0.9, weight_decay=0.0001
        )
        lr_scheduler = WarmupMultiStepLR(
            optimizer, milestones=[240000, 320000]
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
            "monitor": "val_entropy"
        }

check_best=ModelCheckpoint(
    save_top_k=1,
    monitor="val_entropy",
    mode="min",
    dirpath="./lightning_logs/ckpt",
    filename="3-bdd-fast-{epoch:02d}-{ratio:02d}-{val_entropy:.2f}"
)
  
    
def train():

    generator = torch.Generator()
    generator.manual_seed(123)
    train_path = "./bdd100k/images/100k/train" # 7w images
    val_path = "./bdd100k/images/100k/val" # 1w imgs
    
    dataset = BDD100K(root=train_path, annFile="./bdd100k/labels/det_20/det_train_coco.json", transform=PILToTensor())
    # divide train in to 20% labeled dataset and 80% unlabeled pool dataset
    # train_dataset, valid_dataset, _ = torch.utils.data.random_split(dataset=dataset, lengths=[0.001, 0.001, 0.998], generator=generator) 
    train_dataset, _ = torch.utils.data.random_split(dataset=dataset, lengths=[0.05, 0.95], generator=generator) 
    valid_dataset = BDD100K(root=val_path, annFile="./bdd100k/labels/det_20/det_val_coco.json", transform=PILToTensor())


    train_loader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=False, collate_fn=collection)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=1, shuffle=False, collate_fn=collection)
    
    
    net = fasterrcnn_resnet50_fpn(weights=None, num_classes=11)

    fast_model = LitAutoTrain(model=net)

    trainer = L.Trainer(
        max_epochs=80, 
        accelerator="gpu", 
        callbacks=check_best,
        precision="16-mixed",
        devices=[0], 
        log_every_n_steps=1,
        gradient_clip_val=0.5)
    trainer.fit(model=fast_model, train_dataloaders=train_loader, val_dataloaders=valid_loader)

if __name__ == "__main__":
    train()

