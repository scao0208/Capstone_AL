import torch.utils.data
from src.dataset import BDD100K
from src.trainingModule import WarmupMultiStepLR
from torchvision.transforms import PILToTensor

from torchmetrics.detection.mean_ap import MeanAveragePrecision



import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from typing import Any

from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import torch



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
    
    def training_step(self, batch, batch_idx):
        # Batch
        x, y = batch  # tuple unpacking
        loss_dict = self.model(x, y)
        
        loss = sum(loss for loss in loss_dict.values())
        self.log_dict(loss_dict)
        return loss

    def validation_step(self, batch, batch_idx):
        # Batch
        x, y = batch
        y_hat = self.model(x)
        # calculate MAP
        map = MeanAveragePrecision(box_format="xywh", iou_type="bbox")
        map.update(y_hat, y)
        val_map = map.compute()
        val_map.pop("classes") # corresponding to the cocoapi cocoeval.py line 378 + 379, np.float() -> np.float64() 

        # true_boxes, true_labels = {'boxes':y[0]['boxes']}, {'labels':y[0]['labels']}
        # pred_socres, pred_boxes, pred_labels  = {'scores': y_hat[0]['scores']}, {'boxes':y_hat[0]['boxes']}, {'labels':y_hat[0]['labels']}
        self.log_dict(val_map)
        return val_map

       
    # def test_step(self, batch, batch_idx):
    #      # Batch
    #     # x, y, x_name, y_name = batch
    #     x, y = batch
    #     y_hat = self.model(x)
    #     metric = MeanAveragePrecision(iou_type="bbox")
    #     metric.update(y_hat, y)
    #     test_metric = metric.compute()
    #     test_metric.pop("classes")

    #     self.log_dict(test_metric)
    #     return test_metric

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.model.parameters(), lr=self.lr, momentum=0.9, weight_decay=0.0001
        )
        lr_scheduler = WarmupMultiStepLR(
            optimizer, milestones=[60000, 80000]
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
            "monitor": "map_50"
        }

check_list = ModelCheckpoint(
    save_top_k=1,
    monitor="map_50",
    mode="max",
    dirpath="./lightning_logs/ckpt",
    filename="bdd-fast-{epoch:02d}-{map_50:.2f}"
)
    
def train():

    generator = torch.Generator()
    generator.manual_seed(123)
    train_path = "./bdd100k/images/100k/train" # 7w images
    val_path = "./bdd100k/images/100k/val" # 1w imgs
    
    train_dataset = BDD100K(root=train_path, annFile="./bdd100k/labels/det_20/det_train_coco.json", transform=PILToTensor())
    # divide train in to 5w training dataset and 2w validation dataset, to valid the hyperparameter settings
    # train_dataset, val_dataset = torch.utils.data.random_split(dataset=dataset, lengths=[13/14, 1/14], generator=generator) 

    # test whether the correctness of script
    # train_dataset, valid_dataset, _ = torch.utils.data.random_split(dataset=train_dataset, lengths=[20, 20, train_dataset.__len__() - 40], generator=generator)

    # !! The truly first training  
    # train_dataset, val_dataset, _ = torch.utils.data.random_split(dataset=train_dataset, lengths=[0.15, 0.05, 0.8], generator=generator)

    valid_dataset = BDD100K(root=val_path, annFile="./bdd100k/labels/det_20/det_val_coco.json", transform=PILToTensor())
    # test_dataset, _ = torch.utils.data.random_split(dataset=test_dataset, lengths=[5000, test_dataset.__len__() - 5000], generator=generator) 

    # don't set num_workers, because it will cause the Dataloader errors
    train_loader = DataLoader(dataset=train_dataset, batch_size=2, shuffle=False, collate_fn=collection)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=2, shuffle=False, collate_fn=collection)
    # test_loader = DataLoader(dataset=test_dataset, batch_size=2, shuffle=False, collate_fn=collection)
    
    
    net = fasterrcnn_resnet50_fpn(weights=None, num_classes=11)

    fast_model = LitAutoTrain(model=net)

    trainer = L.Trainer(
        max_epochs=45, 
        accelerator="gpu", 
        callbacks=check_list,
        precision="16-mixed",
        devices=[0,1,2], 
        log_every_n_steps=1,
        gradient_clip_val=0.5)
    
    trainer.fit(model=fast_model, train_dataloaders=train_loader, val_dataloaders=valid_loader)

    # trainer.test(model=fast_model, dataloaders=test_loader)

if __name__ == "__main__":
    train()

