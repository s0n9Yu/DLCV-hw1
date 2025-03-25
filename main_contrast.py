import torch
import os
import torchvision.transforms as transforms
import torch.optim.lr_scheduler
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from dataset import MyDataset
from model_contrast import CustomModel

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

trainingData = MyDataset(path="data/train",
                         transform=train_tfm,
                         training=True,
                         contrasive_image=True)
validation = MyDataset(path="data/val",
                       transform=test_tfm,
                       training=True,
                       contrasive_image=False)
testingdata = MyDataset(path="data/test", transform=test_tfm, training=False)

model = CustomModel()

parameternum = sum(p.numel() for p in model.parameters())
print("#parameters =", parameternum)


device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
batch_size = 32
n_epochs = 100
learning_rate = 3e-5
lambda_contrast = 15
criterion = nn.CrossEntropyLoss()
contrasive_loss_margin = 0.2
contrastive_loss_fn = nn.CosineEmbeddingLoss(margin=contrasive_loss_margin)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
train_loader = DataLoader(trainingData,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=32,
                          pin_memory=True,
                          persistent_workers=True,
                          prefetch_factor=8)
validation_loader = DataLoader(validation, batch_size=batch_size, shuffle=True)
best_acc = 0

if useWandb:
    run = wandb.init(
        # Set the project where this run will be logged
        project="dlcv-hw1",
        # Track hyperparameters and run metadata
        config={
            "seed": myseed,
            "#parameter": parameternum,
            "batch size": batch_size,
            "learning_rate": learning_rate,
            "epochs": n_epochs,
            "lambda_contrast": lambda_contrast,
            "contrasive loss margin": contrasive_loss_margin,
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
    contra_loss = []
    train_accs = []

    for batch in tqdm(train_loader):

        # A batch consists of image data and corresponding labels.
        img1, img2, target, label1, label2 = batch
        img1, img2, target, label1, label2 = img1.to(device), \
            img2.to(device), target.to(device), \
            label1.to(device), label2.to(device)

        # Gradients stored in the parameters in the previous step
        # should be cleared out first.
        optimizer.zero_grad()
        # Compute embeddings
        logit1, emb1 = model(img1, return_features=True)
        logit2, emb2 = model(img2, return_features=True)

        # Compute contrastive loss
        contrasive_loss = lambda_contrast \
            * contrastive_loss_fn(emb1, emb2, target)

        # Calculate the cross-entropy loss.
        # We don't need to apply softmax before computing cross-entropy
        # as it is done automatically.
        loss1 = criterion(logit1, label1)
        loss2 = criterion(logit2, label2)
        loss = loss1 + loss2 + contrasive_loss
        loss.backward()

        # Clip the gradient norms for stable training.
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)

        # Update the parameters with computed gradients.
        optimizer.step()

        # Compute the accuracy for current batch.
        acc = (logit1.argmax(dim=-1) == label1.to(device)).float().mean()

        # Record the loss and accuracy.
        train_loss.append(loss.item())
        train_accs.append(acc)
        contra_loss.append(contrasive_loss.item())

    train_loss = sum(train_loss) / len(train_loss)
    train_acc = sum(train_accs) / len(train_accs)
    contra_loss = sum(contra_loss) / len(contra_loss)
    print("train accuracy = ", train_acc,
          "train loss = ", train_loss,
          "contrasive_loss = ", contra_loss)

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
        best_acc = valid_acc
        torch.save(model, f"checkpoint/{model_name}_best.ckpt")

    if useWandb:
        wandb.log({"train acc": train_acc,
                   "train loss": train_loss,
                   "valid acc": valid_acc,
                   "valid loss": valid_loss,
                   "contrasive loss": contra_loss})


torch.save(model, f"checkpoint/{model_name}.ckpt")
print(f"Save the model to checkpoint/{model_name}.ckpt")
