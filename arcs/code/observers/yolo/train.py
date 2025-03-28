import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.transforms.functional as FT
from tqdm import tqdm 
from torch.utils.data import DataLoader
from tracker import Yolov1
from dataset import VOCDataset
from utils import *
from loss import YoloLoss

seed = 123
torch.manual_seed(seed)

# hyperparameters
LR = 2e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 16

WEIGHT_DECAY = 0
EPOCHS = 100

WORKERS = 2
PIN_MEMORY = True
LOAD_MODEL = False

LOAD_MODEL_FILE = "overfit.pth.tar"
 
IMG_DIR = "data/archive/images"
LABEL_DIR = "data/archive/labels"


class Compose(object):
    def __init__(self, transforms):

        self.transforms = transforms
    
    def __call__(self, img, bboxes):
        for t in self.transforms:
            img, bboxes = t(img), bboxes

        return img, bboxes


transform = Compose([transforms.Resize((448,448)), transforms.ToTensor()])

def train_fn(train_loader, model, optimizer, loss_fn):
    loop = tqdm(train_loader, leave=True)
    mean_loss = [] 

    for batch_idx, (x, y) in enumerate(loop):
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)
        loss = loss_fn(out, y)
        mean_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update progress bar
        loop.set_postfix(loss = loss.item())
    
    print(f"Mean loss was {sum(mean_loss)/len(mean_loss)}")


def main():
    print("Device:", DEVICE)
    model = Yolov1(split_size=7, num_boxes=2, num_classes=20).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    loss_fn = YoloLoss()

    if LOAD_MODEL:
        load_checkpoint(torch.load(LOAD_MODEL_FILE), model, optimizer)
    
    train_dataset = VOCDataset(
        "data/archive/8examples.csv", 
        transform=transform, 
        img_dir=IMG_DIR,
        label_dir=LABEL_DIR
    )

    test_dataset = VOCDataset(
        "data/archive/test.csv", 
        transform=transform, 
        img_dir=IMG_DIR,
        label_dir=LABEL_DIR
    )


    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=False,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=True,
    )

    for epoch in range(EPOCHS):
        if epoch >= 90:
            for x, y in train_loader:
                x = x.to(DEVICE)
                for idx in range(8):
                    bboxes = cellboxes_to_boxes(model(x))
                    bboxes = non_max_suppression(bboxes[idx], iou_threshold=0.5, threshold=0.5)
                    plot_image(x[idx].permute(1,2,0).to("cpu"), bboxes)


        pred_boxes, target_boxes = get_bboxes(train_loader, model, iou_threshold=0.5, threshold=0.5)

        mean_avg_prec = mean_average_precision(pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint")

        print(f"Train mAP: {mean_avg_prec}")
        if mean_avg_prec > 0.9:
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict()
            }
            save_checkpoint(checkpoint, filename=LOAD_MODEL_FILE)

        train_fn(train_loader, model, optimizer, loss_fn)



if __name__ == "__main__":

    main()