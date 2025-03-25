import torch 
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse

from dataset import MyDataset
from model import CustomModel

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True, help="model path")
args = parser.parse_args()

test_tfm = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
testingdata = MyDataset(path="data/test", transform=test_tfm, training=False)
testing_loader = DataLoader(testingdata, batch_size=1, shuffle=True)

device = "cuda"

model = torch.load(args.model).to(device)
model.eval()
prediction = []
with torch.no_grad():
    for img, filename in tqdm(testing_loader):
        filename = os.path.basename(filename[0])
        test_pred = model(img.to(device))
        test_label = np.argmax(test_pred.cpu().data.numpy(), axis=1).squeeze()
        prediction.append({"image_name": filename.split(".")[0], "pred_label": test_label})

df = pd.DataFrame(prediction)
df.to_csv("prediction.csv", index = False)



