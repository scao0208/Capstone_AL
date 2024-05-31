import os
from train import LitAutoTrain
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import torch.utils.data
from src.dataset import BDD100K
from torchvision import transforms
from torchvision.transforms import PILToTensor
from PIL import Image
import matplotlib 
import matplotlib.patches as patches
from torch.utils.data import DataLoader
from torchmetrics.detection import MeanAveragePrecision, IntersectionOverUnion

import matplotlib.pyplot as plt

os.makedirs("outplot", exist_ok=True)

def collection(data):
    images = []
    targes = []
    for sample in data:
        images += [sample[0]]
        targes += [sample[1]]
    return images, targes



train_path = "./bdd100k/images/100k/train"

dataset = BDD100K(root="./bdd100k/images/100k/train", annFile="./bdd100k/labels/det_20/det_train_coco.json", transform=PILToTensor())

label_pool, unlabel_pool = torch.utils.data.random_split(dataset=dataset, lengths=[0.2, 0.8])

# val_dataset = BDD100K(root="./bdd100k/images/100k/val", annFile="./bdd100k/labels/det_20/det_val_coco.json", transform=PILToTensor())
unlabel_loader = DataLoader(dataset=unlabel_pool, batch_size=1, shuffle=True, collate_fn=collection)


net = fasterrcnn_resnet50_fpn(weights=None, num_classes=11)

checkpoint = "/home/siyu/Documents/al/scapstone/lightning_logs/ckpt/3-bdd-fast-epoch=41-map_50=0.76.ckpt"

autotest = LitAutoTrain.load_from_checkpoint(checkpoint_path=checkpoint, model=net)
# choose trained nn.Module
autotest.eval()


# Assuming unlabel_loader has been defined earlier
imgs, labels = next(iter(unlabel_loader))

label = [{k: v.cuda() for k, v in label.items()} for label in labels]

out = net([img.cuda() for img in imgs])

# calculate map
map = MeanAveragePrecision(box_format="xywh", iou_type="bbox")
map.update(out, label)
map_50 = map.compute()["map_50"].item()

# list boxes whose iou > 0.5
iou_50 = list()

transformed_labels = list()

gt_boxes = label[0]["boxes"]
gt_labels = label[0]["labels"]
for box, lbl in zip(gt_boxes, gt_labels):
    transformed_labels.append([{'boxes': box.unsqueeze(0), 'labels': lbl.unsqueeze(0)}])

transformed_pred = list()
pred_boxes = out[0]["boxes"]
pred_labels = out[0]["labels"]
pred_scores = out[0]["scores"]
for box, sce, lbl in zip(pred_boxes, pred_scores, pred_labels):
    transformed_pred.append([{'boxes': box.unsqueeze(0), 'scores': sce.unsqueeze(0), 'labels': lbl.unsqueeze(0)}])

for i in range(len(transformed_labels)):
    for j in range(len(transformed_pred)):
        iou = IntersectionOverUnion(box_format="xywh")
        iou.update(transformed_pred[j], transformed_labels[i])
        iou = iou.compute()
        iou_50.append(iou)


print(f"out:{out}", map_50, iou_50)



# Display the image with predicted bounding boxes and labels
fig, ax = plt.subplots()

# Display the image
ax.imshow(imgs[0].permute(1, 2, 0).cpu())

bbox = out[0].get("boxes") # Get bounding box coordinates
label_id = out[0].get("labels")  # Get predicted label ID
score = out[0].get("scores")  # Get score

# Plot bounding box
for i in range(len(bbox)): 
    box = bbox[i].cpu().detach().numpy()
    score_i = score[i].cpu().detach().numpy()
    scores = "{:.2f}".format(score_i)
    # if iou_50[i]["iou"] >= 0.5:
    if score_i >= 0.5:
        x, y, w, h = box[0], box[1], box[2] - box[0], box[3] - box[1]  # Calculate width and height
        bb = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor="blue", facecolor="none")
        ax.add_patch(bb)
        # Add label and score as text
        ax.text(x, y - 30, f'{label_id[i]}', color='red', fontsize=8, verticalalignment='top')
    else:
        continue

plt.show()
plt.savefig('./outplot/pred_image.png')


# Display the image with true bounding boxes and labels

fig1, ax1 = plt.subplots()

# Display the image
ax1.imshow(imgs[0].permute(1, 2, 0).cpu())

bbox_gt = labels[0].get("boxes") # Get bounding box coordinates
label_id_gt = labels[0].get("labels")  # Get predicted label ID


for i in range(len(bbox_gt)): 
    box = bbox_gt[i].cpu().detach().numpy()
    x, y, w, h = box[0], box[1], box[2] - box[0], box[3] - box[1]  # Calculate width and height
    bb = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor="blue", facecolor="none")
    ax1.add_patch(bb)


    # Add label and score as text
    ax1.text(x, y - 30, f'{label_id[i]}', color='red', fontsize=10, verticalalignment='top')

plt.show()
plt.savefig('./outplot/gt_image.png')


