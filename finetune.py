import os
import torch
import torch.utils.data
import numpy as np
from tqdm import tqdm 
from torchvision import transforms
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from train_part_mAP import LitAutoTrain
from torchvision.transforms import PILToTensor
from src.dataloader import collection
from src.dataset import BDD100K, SampleBatches
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.tensorboard.writer import SummaryWriter
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn

train_path = "./bdd100k/images/100k/train"
data_mean = torch.tensor([0.485, 0.456, 0.406])  # need to find real one later
data_std = torch.tensor([0.229, 0.224, 0.225])  # need to calculate 

normalize = transforms.Normalize(
    mean=list(data_mean),
    std=list(data_std),
)

transform = transforms.Compose([transforms.PILToTensor(), normalize])


print("==> Building model..")
device = "cuda" if torch.cuda.is_available() else "cpu"

checkpoint_path = "./lightning_logs/ckpt/3-bdd-fast-epoch=47-ratio=00-map_50=0.74.ckpt"
checkpoint = torch.load(checkpoint_path, map_location=device)

state_dict = checkpoint['state_dict']
new_state_dict = dict()
for k, v in state_dict.items():
    if k.startswith("model"):
        new_state_dict[k[6:]] = v
    else:
        new_state_dict[k] = v

model = fasterrcnn_resnet50_fpn(weights=None, num_classes=11)
model.load_state_dict(new_state_dict)
model = model.to(device)

def get_plabels2(model, samples, device):
    sub5k_set = SampleBatches(root=train_path, feasible_list=samples, annFile="./bdd100k/labels/det_20/det_train_coco.json", transform=transforms.PILToTensor())
    sub5k = DataLoader(sub5k_set, batch_size=8, shuffle=False, collate_fn=collection, pin_memory=True)
    
    top1_scores = []
    model.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets, paths, indices) in enumerate(tqdm(sub5k, desc=f'getting labels on {device}', colour='yellow')):
            inputs = [input.to(device) for input in inputs]
            targets = [{k:v.to(device) for k, v in target.items()} for target in targets]
            outputs = model(inputs)
            map_metric = MeanAveragePrecision(box_format='xywh', iou_type='bbox')

            for i in range(len(inputs)):
                list_target = [targets[i]]
                output = [outputs[i]]
                map_metric.update(output, list_target)
                map_50 = map_metric.compute()["map_50"].item()
                top1_scores.append(map_50)

    model.train()
    idx = np.argsort(top1_scores)
    samples = np.array(samples)
    return samples[idx[:1000]]


if __name__ == "__main__":
    generator = torch.Generator()
    generator.manual_seed(123)
    cycle = 1
    labeled = list()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    trainset = SampleBatches(root=train_path, feasible_list=labeled, annFile="./bdd100k/labels/det_20/det_train_coco.json", transform=transforms.PILToTensor())
    logger = TensorBoardLogger("lightning_logs", name="v2", version="version_0")
    writer = SummaryWriter(log_dir=logger.log_dir)

    fast_model = LitAutoTrain(model=model)

    val_path = "./bdd100k/images/100k/val"
    valid_dataset = BDD100K(root=val_path, annFile="./bdd100k/labels/det_20/det_val_coco.json", transform=PILToTensor())
    # valid_dataset, _ = torch.utils.data.random_split(valid_dataset, lengths=[10, len(valid_dataset) - 10])
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=8, shuffle=False, collate_fn=collection)

    print('Cycle', cycle)
    
    with open(f'./mAP50_id_v2/batch_0.txt', 'r') as f:
        samples = f.readlines()
        samples = [int(samples[i].strip("\n")) for i in range(len(samples))]
        # samples = [int(samples[i].strip("\n")) for i in range(500)]
        samples1k = get_plabels2(model, samples, device)
    
    labeled.extend(samples1k)
    print(f'>> Labelecd length: {len(labeled)}')
    
    trainset.feasible_list = labeled
    train_loader = DataLoader(trainset, batch_size=8, shuffle=False, collate_fn=collection, pin_memory=True)
    
    check_best = ModelCheckpoint(
        save_top_k=1,
        monitor="map_50",
        mode="max",
        dirpath=logger.log_dir + f"/checkpoint_v2/cycle_{0}",
        filename=f"4-sample-cycle={cycle:02d}" + "-{epoch:02d}-{map_50:.2f}"
    )

    trainer = L.Trainer(
        max_epochs=120,
        callbacks=[check_best],
        logger=logger,
        precision="16-mixed",
        devices=[0],
        log_every_n_steps=1,
        gradient_clip_val=0.5
    )

    trainer.fit(model=fast_model, train_dataloaders=train_loader, val_dataloaders=valid_loader)
    trainer.fit_loop.epoch_progress.current.completed = 0

    writer.close()
