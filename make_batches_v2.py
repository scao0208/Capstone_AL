import os
import numpy as np
import torch.utils
# import wandb
import logging

import torch
import torch.utils.data

from torchvision import transforms
from src.dataset import BDD100K
from src.dataloader import collection
from torchmetrics.detection import MeanAveragePrecision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm



device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
train_path = "/home/sycao/Documents/scapstone/bdd100k/images/100k/train/"
# checkpoint_path = "./lightning_logs/ckpt/3-bdd-fast-epoch=59-ratio=00-map_50=0.71.ckpt"
checkpoint_path = "./lightning_logs/ckpt/3-bdd-fast-epoch=47-ratio=00-map_50=0.74.ckpt"


print('==> Loading data...')

generator = torch.Generator()
generator.manual_seed(123)
trainset = BDD100K(root=train_path, annFile="/home/sycao/Documents/scapstone/bdd100k/labels/det_20/det_train_coco.json", transform=transforms.PILToTensor())

# For test   
_, trainset= torch.utils.data.random_split(dataset=trainset, lengths=[0.7, 0.3], generator=generator)
 
# label_pool, unlabel_pool, _ = torch.utils.data.random_split(dataset=trainset, lengths=[20, 10, trainset.__len__() - 30], generator=generator)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=8, shuffle=False, collate_fn=collection, pin_memory=True)

os.makedirs("tensor_logs", exist_ok=True)

# logging
log_file = f"tensor_logs/split.log"
logging.basicConfig(level=logging.INFO, filename=log_file, filemode='w')
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger('').addHandler(console)
# fix random seeds
# torch.use_deterministic_algorithms(True)
writer = SummaryWriter(log_dir="tensor_logs/")

# use the fasterrcnn backbone with its object detection header


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

# if device == 'cuda':
#     model = nn.DataParallel(model, device_ids=[0,1,2,3])
#     cudnn.benchmark = True



# First we do the inference for all training data find the pictures with low mAP50 
def inference():
    global best_acc
    model.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets, paths, indices) in enumerate(tqdm(train_loader, desc='Validating', colour='yellow')):
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
                # entropy = sum(- scores * torch.log(scores)).mean()
                entropy = sum(- scores * torch.log(scores))
                
                map.update(output, list_target)
                map_50 = map.compute()["map_50"].item()
                mar_large = map.compute()["mar_large"].item()
                writer.add_scalar("mAP50", map_50, batch_idx)
                writer.add_scalar("mAR_Large", mar_large, batch_idx)

                s = str(float(map_50)) + '_' + str(img_id[0]) + '_' + str(paths[i]) + "\n"

                with open('./inference_mAP50_v2.txt', 'a') as f:
                    f.write(s)
    

            # wandb.log({"1_main_test_loss": test_loss/(batch_idx + 1), "1_main_test_acc": 100.*correct/total})
        

if __name__ == "__main__":
    inference()
    writer.close()

    with open('./inference_mAP50_v2.txt', 'r') as f:
        scores = f.readlines()
    
    score_1 = []
    name_2 = []
    path_3 = []
    
    for j in scores:
        score_1.append(j[:-1].split('_')[0])
        name_2.append(j[:-1].split('_')[1])
        path_3.append(j[:-1].split('_')[2])
    
    s = np.array(score_1)
    sort_index = np.argsort(s)
    
    x = sort_index.tolist()
    
    sort_index = np.array(x) # convert to low score first
    
    # Now we get all data sorted in the order of low accuracy 
    if not os.path.isdir('first_mAP50_v2'):
        os.mkdir('first_mAP50_v2')

    s = './first_mAP50_v2/batch_0.txt'
    with open(s, 'a') as f:
        for k in range(len(name_2)):
            f.write(name_2[k] + '_' + path_3[k] + '\n')

    
    
    
    