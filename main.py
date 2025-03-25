import torch
import os
import torchvision.transforms as transforms
import torch.optim.lr_scheduler
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from dataset import MyDataset
from model import CustomModel

import wandb
if os.environ.get('WANDB_API_KEY') is not None:
    useWandb = True
    wandb.login()
else:
    useWandb = False

# reproduction
myseed = 6666  # set a random seed for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(myseed)
torch.manual_seed(myseed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(myseed)


train_tfm = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(50),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomAdjustSharpness(sharpness_factor=2),
    transforms.ToTensor(),
])

test_tfm = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

trainingData = MyDataset(path="data/train", transform=train_tfm, training=True)
validation = MyDataset(path="data/val", transform=test_tfm, training=True)
testingdata = MyDataset(path="data/test", transform=test_tfm, training=False)

model = CustomModel()

'''
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', weights=True)
# Freeze the pretrained model
for param in model.parameters():
    param.requires_grad = False
#model.fc = nn.Linear(in_features = 512, out_features = 100, bias = True)
model.fc = nn.Sequential(
    nn.Linear(in_features=512, out_features=64),
    nn.BatchNorm1d(64),
    nn.Dropout(p = 0.5),
    nn.Linear(in_features=64, out_features=64),
    nn.BatchNorm1d(64),
    nn.Dropout(p = 0.5),
    nn.Linear(in_features=64, out_features=100)
)
'''
parameternum = sum(p.numel() for p in model.parameters())
print("#parameters =", parameternum)


device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
batch_size = 64
n_epochs = 50
learning_rate = 3e-5
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
train_loader = DataLoader(trainingData,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=32,
                          pin_memory=True,
                          persistent_workers=True,
                          prefetch_factor=2)
validation_loader = DataLoader(validation, batch_size=batch_size, shuffle=True)
best_acc = 0

if useWandb:
    run = wandb.init(
        # Set the project where this run will be logged
        project="dlcv-hw1",
        # Track hyperparameters and run metadata
        config={
            "#parameter": parameternum,
            "batch size": batch_size,
            "learning_rate": learning_rate,
            "epochs": n_epochs,
            "model": str(model),
        },
        notes="",
    )
    model_name = wandb.run.name
else:
    model_name = "model"

os.makedirs("checkpoint", exist_ok=True)
for epoch in range(n_epochs):

    # ---------- Training ----------
    # Make sure the model is in train mode before training.
    model.train()

    # These are used to record information in training.
    train_loss = []
    train_accs = []

    for batch in tqdm(train_loader):

        # A batch consists of image data and corresponding labels.
        imgs, labels = batch

        # Forward the data. (Make sure data and model are on the same device.)
        logits = model(imgs.to(device))

        # Calculate the cross-entropy loss.
        # We don't need to apply softmax before computing cross-entropy
        # as it is done automatically.
        loss = criterion(logits, labels.to(device))

        # Gradients stored in the parameters in the previous step
        # should be cleared out first.
        optimizer.zero_grad()

        # Compute the gradients for parameters.
        loss.backward()

        # Clip the gradient norms for stable training.
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)

        # Update the parameters with computed gradients.
        optimizer.step()

        # Compute the accuracy for current batch.
        acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()

        # Record the loss and accuracy.
        train_loss.append(loss.item())
        train_accs.append(acc)

    train_loss = sum(train_loss) / len(train_loss)
    train_acc = sum(train_accs) / len(train_accs)
    print("train accuracy = ", train_acc, "train loss = ", train_loss)

    model.eval()

    # These are used to record information in validation.
    valid_loss = []
    valid_accs = []

    # Iterate the validation set by batches.
    for batch in tqdm(validation_loader):

        # A batch consists of image data and corresponding labels.
        imgs, labels = batch

        # We don't need gradient in validation.
        # Using torch.no_grad() accelerates the forward process.
        with torch.no_grad():
            logits = model(imgs.to(device))

        # We can still compute the loss (but not the gradient).
        loss = criterion(logits, labels.to(device))

        # Compute the accuracy for current batch.
        acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()

        # Record the loss and accuracy.
        valid_loss.append(loss.item())
        valid_accs.append(acc)

    # The average loss and accuracy for entire validation set
    # is the average of the recorded values.
    valid_loss = sum(valid_loss) / len(valid_loss)
    valid_acc = sum(valid_accs) / len(valid_accs)
    print("valid accuracy = ", valid_acc, "valid loss = ", valid_loss)

    if valid_acc > best_acc:
        torch.save(model, f"checkpoint/{model_name}_best.ckpt")

    if useWandb:
        wandb.log({"train acc": train_acc,
                   "train loss": train_loss,
                   "valid acc": valid_acc,
                   "valid loss": valid_loss})

torch.save(model, f"checkpoint/{model_name}.ckpt")
print(f"Save the model to checkpoint/{model_name}.ckpt")
