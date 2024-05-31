import os
import torch
import torch.nn.functional as F
import torch.utils.data
import numpy as np
from tqdm import tqdm 
import glob
from torchvision import transforms
import lightning as L
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from train_part import LitAutoTrain
from torchvision.transforms import PILToTensor
from src.dataloader import collection
from src.dataset import BDD100K, SampleBatches
from lightning.pytorch.callbacks import ModelCheckpoint
from typing import Any
from torch.utils.tensorboard.writer import SummaryWriter

from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn


data_mean = torch.tensor([0.485, 0.456, 0.406]) # need to find real one later
data_std = torch.tensor([0.229, 0.224, 0.225]) # need to calculate 

normalize = transforms.Normalize(
            mean=list(data_mean),
            std=list(data_std),
        )

transform = transforms.Compose([transforms.PILToTensor(), 
                                 normalize])


device = 'cuda' if torch.cuda.is_available() else 'cpu'
start_epoch = 0  # start from epoch 0 or last checkpoint epoch


print("==> Building model..")
network = fasterrcnn_resnet50_fpn(weights=None, num_classes=11)

# class Sample_Train(LitAutoTrain):
#     def __init__(self, model, lr: float = 0.01, *args: Any, **kwargs: Any) -> None:
#         super().__init__(*args, **kwargs)
#         self.model= model
#         self.lr = lr
#         # outputs
#         self.training_step_outputs = []
#         self.validation_step_outputs = []
#         self.test_step_outputs = []
    
#     def training_step():


#     acc = 100.*correct/total
#         if acc > best_acc:
#             print('Saving..')
#             state = {
#                 'net': net.state_dict(),
#                 'acc': acc,
#                 'epoch': epoch,
#             }
#             if not os.path.isdir('checkpoint'):
#                 os.mkdir('checkpoint')
#             torch.save(state, f'./checkpoint/main_{cycle}.pth')
#             best_acc = acc

def get_plabels2(model, samples):

    sub5k_set = SampleBatches(root=train_path, feasible_list=samples, annFile="/home/siyu/Documents/al/scapstone/bdd100k/labels/det_20/det_train_coco.json", transform=transforms.PILToTensor())
    sub5k_set, _ = torch.utils.data.random_split(sub5k_set, lengths=[100, len(sub5k_set) - 100])
    sub5k = DataLoader(sub5k_set, batch_size=4, shuffle=False, collate_fn=collection, pin_memory=True)
    

    # entropy_indicators = list()
    top1_scores = list()
    model.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets, paths, indices) in enumerate(tqdm(sub5k, desc='getting labels', colour='yellow')):
            # inputs, targets = inputs.to(device), targets.to(device)
            inputs = [input.cuda() for input in inputs]
            targets = [{k:v.cuda() for k, v in target.items()} for target in targets]
            outputs = model(inputs)
            map = MeanAveragePrecision(box_format='xywh', iou_type='bbox')

            for i in range(len(inputs)):
                list_target = [targets[i]]

                output = [outputs[i]]
                img_id = [indices[i]]

                num_preds = len(output[0]["boxes"])
                scores = output[0]["scores"]
                # entropy = sum(- scores * torch.log(scores) - (1 - scores) * torch.log(1 - scores)).mean()
                entropy = sum(- scores * torch.log(scores))
                # entropy = sum(- scores * torch.log(scores)).mean()
                
                map.update(output, list_target)
                map_50 = map.compute()["map_50"].item()
                mar_large = map.compute()["mar_large"].item()
            top1_scores.append(map_50)
            # entropy_indicators.append(entropy)

    idx = np.argsort(top1_scores)
    samples = np.array(samples)
    return samples[idx[:10]]



    
if __name__ == "__main__":
    
    generator = torch.Generator()
    generator.manual_seed(123)
    CYCLE = 10
    labeled = list()
    logger = SummaryWriter(log_dir='./lightning_logs/')
    for cycle in range(CYCLE):    
        print('Cycle', cycle)

        with open(f'/home/siyu/Documents/al/scapstone/mAP50_id/batch_{cycle}.txt', 'r') as f:
            samples = f.readlines()
            samples = [int(samples[i].strip("\n")) for i in range(len(samples))]
            
        if cycle > 0:
            print('>> Getting previous checkpoint')
            checkpoint = glob.glob(f'/home/siyu/Documents/al/scapstone/lightning_logs/checkpoint/cycle_{cycle-1}/4-sample-cycle={cycle-1:02d}-*.ckpt')
            ck_path = torch.load(checkpoint[0], map_location=device)
            state_dict = ck_path['state_dict']
            new_state_dict = dict()
            for k, v in state_dict.items():
                if k.startswith("model"):
                    new_state_dict[k[6:]] = v
                else:
                    new_state_dict[k] = v

            model = fasterrcnn_resnet50_fpn(weights=None, num_classes=11)

            model.load_state_dict(new_state_dict)
            model = model.to(device)
            # sampling
            sample1k = get_plabels2(model, samples)

            lit_model = LitAutoTrain.load_from_checkpoint(checkpoint_path=checkpoint[0], model=network)
            fast_model = LitAutoTrain(model=lit_model)          
        else:
            # first iteration: sample 1k at even intervals
            samples = np.array(samples)
            sample1k = samples[[j*50 for j in range(100)]]
            fast_model = LitAutoTrain(model=network)
        # add 1k samples to labeled set
        labeled.extend(sample1k)
        print(f'>> Labeled length: {len(labeled)}')

        train_path = "./bdd100k/images/100k/train"

        trainset = SampleBatches(root=train_path, feasible_list=labeled, annFile="/home/siyu/Documents/al/scapstone/bdd100k/labels/det_20/det_train_coco.json", transform=transforms.PILToTensor())
        train_loader = DataLoader(trainset, batch_size=4, shuffle=False, collate_fn=collection, pin_memory=True)
   

        val_path = "./bdd100k/images/100k/val" # 1w imgs
        valid_dataset = BDD100K(root=val_path, annFile="./bdd100k/labels/det_20/det_val_coco.json", transform=PILToTensor())
        valid_dataset, _ = torch.utils.data.random_split(valid_dataset, lengths=[100, len(valid_dataset) - 100])
        valid_loader = DataLoader(dataset=valid_dataset, batch_size=2, shuffle=False, collate_fn=collection)
        
        check_best=ModelCheckpoint(
            save_top_k=1,
            monitor="map_50",
            mode="max",
            dirpath=f"./lightning_logs/checkpoint/cycle_{cycle}",
            filename=f"4-sample-cycle={cycle:02d}"+"-{epoch:02d}-{map_50:.2f}"
        )
    
        trainer = L.Trainer(
            max_epochs=2,
            accelerator="gpu",
            callbacks=[check_best],
            precision="16-mixed",
            devices=[0],
            log_every_n_steps=1,
            gradient_clip_val=0.5)
        trainer.fit(model=fast_model, train_dataloaders=train_loader, val_dataloaders=valid_loader)

     
