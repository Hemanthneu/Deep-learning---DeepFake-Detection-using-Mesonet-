
from google.colab import drive
drive.mount('/content/gdrive',force_remount=True)

"""**Load Torch as well as some python libraries**"""

import torch
import numpy as np
from tqdm.auto import tqdm
import pandas as pd
import matplotlib.pyplot as plt
print(torch.__version__)
import torchvision
from torch import nn

# torch vision libraries
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

# Get some data
import zipfile
from pathlib import Path

# Setup a data folder
data_path= Path("data/")
image_path= data_path/"deepfake_image"


if image_path.is_dir():
  print(f"{image_path} skipping download")
else:
  print("downloading the file")
  image_path.mkdir(parents=True, exist_ok=True)

with zipfile.ZipFile("/content/gdrive/MyDrive/data.zip", "r") as zipRef:
  print("Unzipping the zip file")
  zipRef.extractall(image_path)

# Setup train and testing paths
train_dir= image_path/"/content/data/deepfake_image/content/real_vs_fake/real_fake/train"
test_dir= image_path/"/content/data/deepfake_image/content/real_vs_fake/real_fake/test"

train_dir, test_dir

import random
from PIL import Image
random.seed(42)
image_path_list= list(image_path.glob("*/*/*/*/*/*.jpg"))
random_image_path= random.choice(image_path_list)
image_class= random_image_path.parent.stem
img= Image.open(random_image_path)
print(f"Random image path: {random_image_path}")
print(f"Image class: {image_class}")
print(f"Image height: {img.height}")
print(f"Image width: {img.width}")

# use matplotlib to visualize the data
import numpy as np
import matplotlib.pyplot as plt

img_as_array= np.asarray(img)
plt.figure(figsize=(10,5))
plt.imshow(img_as_array)
plt.title(f"Image class: {image_class}")
plt.axis(False)

from torchvision import transforms

data_transfrom= transforms.Compose([
    transforms.Resize(size=(64,64)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor()
])

"""**Visualizing a part of the dataset**"""

def transform_image(image_paths, transform, n=3, seed=None):
  random.seed(seed)
  random_image_paths= random.sample(image_paths, k=n)
  print(random_image_paths)
  for image_path in random_image_paths:
    with Image.open(image_path) as f:
      fig,ax= plt.subplots(1,2)
      ax[0].imshow(f)
      ax[0].set_title(f"Original\nSize: {f.size}")
      ax[0].axis(False)
      transformed_image= transform(f).permute(1, 2, 0)
      ax[1].imshow(transformed_image)
      ax[1].set_title(f"Transformed\nshape : {transformed_image.shape}")
      ax[1].axis(False)

      fig.suptitle(f"Class: {image_path.parent.stem}", fontsize=16)

transform_image(image_path_list, data_transfrom,n=3, seed=42)

from torchvision import datasets
train_data= datasets.ImageFolder(root=train_dir,
                                 transform= data_transfrom,
                                 target_transform=None)
test_data= datasets.ImageFolder(root=test_dir,
                                 transform= data_transfrom,
                                 target_transform=None)

(train_data, test_data)

BATCH_SIZE= 32
train_dataloader= DataLoader( dataset= train_data,
                             batch_size=BATCH_SIZE, shuffle=True)
test_dataloader= DataLoader( dataset= test_data,
                             batch_size=BATCH_SIZE, shuffle=False)

len(train_dataloader),test_dataloader.dataset

"""**Normalizing the images to make it easier for the model to learn the weights**"""

data_transfrom = transforms.Compose([
    transforms.Resize(64),
    transforms.CenterCrop(64),
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.5207, 0.4260, 0.3808],
                         std= [0.2716, 0.2460, 0.2472])
])

def accuracy_fn(y_true, y_pred):
  correct = torch.eq(y_true, y_pred).sum().item()
  acc = (correct / len(y_pred)) * 100
  return acc

from torch.nn.modules.activation import LeakyReLU
class MesoNet(nn.Module):
	def __init__(self,input_features:int, output_features:int):
		super().__init__()

		self.convBlock1 = nn.Sequential(
				nn.Conv2d(in_channels=3,out_channels=100, kernel_size=3, padding=1, bias=False),
				nn.BatchNorm2d(100),
				nn.ReLU(inplace=True),
				nn.LeakyReLU(0.1),
				nn.Conv2d(in_channels=100, out_channels=250, kernel_size=3, padding=1, bias=False),
				nn.BatchNorm2d(250),
				nn.ReLU(inplace=True),
				nn.LeakyReLU(0.1),
				nn.Conv2d(in_channels=250, out_channels=250, kernel_size=3, padding=1, bias=False),
				nn.BatchNorm2d(250),
				nn.ReLU(inplace=True),
				nn.LeakyReLU(0.1),
				nn.Conv2d(in_channels=250, out_channels=250, kernel_size=3, padding=1, bias=False),
				nn.BatchNorm2d(250),
				nn.ReLU(inplace=True),
				nn.LeakyReLU(0.1),
				nn.Conv2d(in_channels=250, out_channels=8, kernel_size=3, padding=1, bias=False),
				nn.BatchNorm2d(8),
				nn.ReLU(inplace=True),
				nn.LeakyReLU(0.1),
				nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, padding=1, bias=False),
				nn.BatchNorm2d(8),
				nn.ReLU(inplace=True),
				nn.LeakyReLU(0.1)
		)
	
		self.convBlock2 = nn.Sequential(
				nn.Conv2d(in_channels=8, out_channels=8, kernel_size=5, padding=2, bias=False),
				nn.BatchNorm2d(8),
				nn.Conv2d(in_channels=8, out_channels=16, kernel_size=5, padding=2, bias=False),
				nn.Conv2d(in_channels=16, out_channels=16, kernel_size=5, padding=2, bias=False),
				nn.MaxPool2d(kernel_size=2),
				nn.MaxPool2d(kernel_size=4)
		)

		self.dropout = nn.Dropout2d(0.5)
		self.fc1 = nn.Linear(1024, 16)
		self.fc2 = nn.Linear(16, output_features)
		self.leakRelu= nn.LeakyReLU(0.1)

	def forward(self,x):
		x = self.convBlock1(x)
		x = self.convBlock2(x)
		x = x.view(x.size(0),-1)
		x = self.dropout(x)
		x = self.fc1(x)
		x = self.leakRelu(x)
		x=  self.dropout(x)
		x = self.fc2(x)
	
		return x

class MesoInception4(nn.Module):
	def __init__(self,input_features:int, output_features:int):
		super().__init__()

    #InceptionLayer1
		self.Inc1_conv1 = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1, padding=0, bias=False)
		self.Inc1_conv2_1 = nn.Conv2d(in_channels=3, out_channels=4, kernel_size=1, padding=0, bias=False)
		self.Inc1_conv2_2 = nn.Conv2d(in_channels=4, out_channels=4, kernel_size=3, padding=1, bias=False)
		self.Inc1_conv3_1 = nn.Conv2d(in_channels=3, out_channels=4, kernel_size=1, padding=0, bias=False)
		self.Inc1_conv3_2 = nn.Conv2d(in_channels=4, out_channels=4, kernel_size=3, padding=2, dilation=2, bias=False)
		self.Inc1_conv4_1 = nn.Conv2d(in_channels=3, out_channels=2, kernel_size=1, padding=0, bias=False)
		self.Inc1_conv4_2 = nn.Conv2d(in_channels=2, out_channels=2, kernel_size=3, padding=3, dilation=3, bias=False)
		self.Inc1_b_n = nn.BatchNorm2d(11)
  
    #InceptionLayer2
		self.Inc2_conv1 = nn.Conv2d(in_channels=11, out_channels=2, kernel_size=1, padding=0, bias=False)
		self.Inc2_conv2_1 = nn.Conv2d(in_channels=11, out_channels=4, kernel_size=1, padding=0, bias=False)
		self.Inc2_conv2_2 = nn.Conv2d(in_channels=4, out_channels=4, kernel_size=3, padding=1, bias=False)
		self.Inc2_conv3_1 = nn.Conv2d(in_channels=11, out_channels=4, kernel_size=1, padding=0, bias=False)
		self.Inc2_conv3_2 = nn.Conv2d(in_channels=4, out_channels=4, kernel_size=3, padding=2, dilation=2, bias=False)
		self.Inc2_conv4_1 = nn.Conv2d(in_channels=11, out_channels=2, kernel_size=1, padding=0, bias=False)
		self.Inc2_conv4_2 = nn.Conv2d(in_channels=2, out_channels=2, kernel_size=3, padding=3, dilation=3, bias=False)
		self.Inc2_b_n = nn.BatchNorm2d(12)
  
    #Normal Layer
		self.conv1 = nn.Conv2d(in_channels=12, out_channels=16, kernel_size=5, padding=2, bias=False)
		self.relu = nn.ReLU(inplace=True)
		self.leakyrelu = nn.LeakyReLU(0.1)
		self.batchnorm1 = nn.BatchNorm2d(16)
		self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2))
		self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=5, padding=2, bias=False)
		self.maxpool2 = nn.MaxPool2d(kernel_size=(4, 4))
		self.dropout = nn.Dropout2d(0.5)
		self.fc1 = nn.Linear(64, 16)
		self.fc2 = nn.Linear(16, output_features)
  
  #InceptionLayer
	def Inception_Layer_1(self, input):
		x1 = self.Inc1_conv1(input)
		x2 = self.Inc1_conv2_1(input)
		x2 = self.Inc1_conv2_2(x2)
		x3 = self.Inc1_conv3_1(input)
		x3 = self.Inc1_conv3_2(x3)
		x4 = self.Inc1_conv4_1(input)
		x4 = self.Inc1_conv4_2(x4)
		y = torch.cat((x1, x2, x3, x4), 1)
		y = self.Inc1_b_n(y)
		y = self.maxpool1(y)
		return y

	def Inception_Layer_2(self, input):
		x1 = self.Inc2_conv1(input)
		x2 = self.Inc2_conv2_1(input)
		x2 = self.Inc2_conv2_2(x2)
		x3 = self.Inc2_conv3_1(input)
		x3 = self.Inc2_conv3_2(x3)
		x4 = self.Inc2_conv4_1(input)
		x4 = self.Inc2_conv4_2(x4)
		y = torch.cat((x1, x2, x3, x4), 1)
		y = self.Inc2_b_n(y)
		y = self.maxpool1(y)
		return y

	def forward(self, input):
		x = self.Inception_Layer_1(input) 
		x = self.Inception_Layer_2(x) 
		x = self.conv1(x) 
		x = self.relu(x)
		x = self.batchnorm1(x)
		x = self.maxpool1(x) 
		x = self.conv2(x)
		x = self.relu(x)
		x = self.batchnorm1(x)
		x = self.maxpool2(x)
		x = x.view(x.size(0), -1) 
		x = self.dropout(x)
		x = self.fc1(x) 
		x = self.leakyrelu(x)
		x = self.dropout(x)
		x = self.fc2(x)
		return x

# Inverted Bottleneck Block
class IB_Block(nn.Module):
    def __init__(self, input_channels, reduction_ratio=16):
        super(IB_Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(input_channels, input_channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(input_channels // reduction_ratio, input_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

from torch.nn.modules.activation import LeakyReLU
class MesoNet_Attention1(nn.Module):
  def __init__(self,input_features:int, output_features:int):
    super().__init__()

    self.convBlock1 = nn.Sequential(
        nn.Conv2d(in_channels=3,out_channels=100, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(100),
        nn.ReLU(inplace=True),
        nn.LeakyReLU(0.1),
        nn.Conv2d(in_channels=100, out_channels=250, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(250),
        nn.ReLU(inplace=True),
        nn.LeakyReLU(0.1),
        nn.Conv2d(in_channels=250, out_channels=250, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(250),
        nn.ReLU(inplace=True),
        nn.LeakyReLU(0.1),
        nn.Conv2d(in_channels=250, out_channels=250, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(250),
        nn.ReLU(inplace=True),
        nn.LeakyReLU(0.1),
        nn.Conv2d(in_channels=250, out_channels=8, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(8),
        nn.ReLU(inplace=True),
        nn.LeakyReLU(0.1),
        nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(8),
        nn.ReLU(inplace=True),
        nn.LeakyReLU(0.1)
    )
    
    self.ib_block = IB_Block(input_channels=8)


    self.AttBlock = nn.Sequential(
        nn.Conv2d(in_channels=8, out_channels=8, kernel_size=1, padding=0, bias=False),
        nn.Sigmoid()
    )

    self.convBlock2 = nn.Sequential(
        nn.Conv2d(in_channels=8, out_channels=8, kernel_size=5, padding=2, bias=False),
        nn.BatchNorm2d(8),
        nn.Conv2d(in_channels=8, out_channels=16, kernel_size=5, padding=2, bias=False),
        nn.Conv2d(in_channels=16, out_channels=16, kernel_size=5, padding=2, bias=False),
        nn.MaxPool2d(kernel_size=2),
        nn.MaxPool2d(kernel_size=4)
    )

    self.dropout = nn.Dropout2d(0.5)
    self.fc1 = nn.Linear(1024, 16)
    self.fc2 = nn.Linear(16, output_features)
    self.leakRelu= nn.LeakyReLU(0.1)

  def forward(self,x):
    x = self.convBlock1(x)
    # Comment this line for 1*1 attention block
    x = self.ib_block(x)
    y = self.AttBlock(x)
    x = torch.mul(x,y)
    x = self.convBlock2(x)
    x = x.view(x.size(0),-1)
    x = self.dropout(x)
    x = self.fc1(x)
    x = self.leakRelu(x)
    x=  self.dropout(x)
    x = self.fc2(x)

    return x

class MesoInception4_Attention1(nn.Module):
	def _init_(self,input_features:int, output_features:int):
		super()._init_()

    #InceptionLayer1
		self.Inc1_conv1 = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1, padding=0, bias=False)
		self.Inc1_conv2_1 = nn.Conv2d(in_channels=3, out_channels=4, kernel_size=1, padding=0, bias=False)
		self.Inc1_conv2_2 = nn.Conv2d(in_channels=4, out_channels=4, kernel_size=3, padding=1, bias=False)
		self.Inc1_conv3_1 = nn.Conv2d(in_channels=3, out_channels=4, kernel_size=1, padding=0, bias=False)
		self.Inc1_conv3_2 = nn.Conv2d(in_channels=4, out_channels=4, kernel_size=3, padding=2, dilation=2, bias=False)
		self.Inc1_conv4_1 = nn.Conv2d(in_channels=3, out_channels=2, kernel_size=1, padding=0, bias=False)
		self.Inc1_conv4_2 = nn.Conv2d(in_channels=2, out_channels=2, kernel_size=3, padding=3, dilation=3, bias=False)
		self.Inc1_b_n = nn.BatchNorm2d(11)
	
		self.AttBlock = nn.Conv2d(in_channels=11, out_channels=11, kernel_size=1, padding=0, bias=False)
		self.sig = nn.Sigmoid()
  
    #InceptionLayer2
		self.Inc2_conv1 = nn.Conv2d(in_channels=11, out_channels=2, kernel_size=1, padding=0, bias=False)
		self.Inc2_conv2_1 = nn.Conv2d(in_channels=11, out_channels=4, kernel_size=1, padding=0, bias=False)
		self.Inc2_conv2_2 = nn.Conv2d(in_channels=4, out_channels=4, kernel_size=3, padding=1, bias=False)
		self.Inc2_conv3_1 = nn.Conv2d(in_channels=11, out_channels=4, kernel_size=1, padding=0, bias=False)
		self.Inc2_conv3_2 = nn.Conv2d(in_channels=4, out_channels=4, kernel_size=3, padding=2, dilation=2, bias=False)
		self.Inc2_conv4_1 = nn.Conv2d(in_channels=11, out_channels=2, kernel_size=1, padding=0, bias=False)
		self.Inc2_conv4_2 = nn.Conv2d(in_channels=2, out_channels=2, kernel_size=3, padding=3, dilation=3, bias=False)
		self.Inc2_b_n = nn.BatchNorm2d(12)
  
    #Normal Layer
		self.conv1 = nn.Conv2d(in_channels=12, out_channels=16, kernel_size=5, padding=2, bias=False)
		self.relu = nn.ReLU(inplace=True)
		self.leakyrelu = nn.LeakyReLU(0.1)
		self.batchnorm1 = nn.BatchNorm2d(16)
		self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2))
		self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=5, padding=2, bias=False)
		self.maxpool2 = nn.MaxPool2d(kernel_size=(4, 4))
		self.dropout = nn.Dropout2d(0.5)
		self.fc1 = nn.Linear(64, 16)
		self.fc2 = nn.Linear(16, output_features)
  
  
  #InceptionLayer
	def Inception_Layer_1(self, input):
		x1 = self.Inc1_conv1(input)
		x2 = self.Inc1_conv2_1(input)
		x2 = self.Inc1_conv2_2(x2)
		x3 = self.Inc1_conv3_1(input)
		x3 = self.Inc1_conv3_2(x3)
		x4 = self.Inc1_conv4_1(input)
		x4 = self.Inc1_conv4_2(x4)
		y = torch.cat((x1, x2, x3, x4), 1)
		y = self.Inc1_b_n(y)
		y = self.maxpool1(y)
		return y

	def Inception_Layer_2(self, input):
		x1 = self.Inc2_conv1(input)
		x2 = self.Inc2_conv2_1(input)
		x2 = self.Inc2_conv2_2(x2)
		x3 = self.Inc2_conv3_1(input)
		x3 = self.Inc2_conv3_2(x3)
		x4 = self.Inc2_conv4_1(input)
		x4 = self.Inc2_conv4_2(x4)
		y = torch.cat((x1, x2, x3, x4), 1)
		y = self.Inc2_b_n(y)
		y = self.maxpool1(y)
		return y


	def forward(self, input):
		x = self.Inception_Layer_1(input) 
		x1 = self.AttBlock(x)
		x1 = self.sig(x1)
		x = torch.mul(x,x1)
		x = self.Inception_Layer_2(x) 
		x = self.conv1(x) 
		x = self.relu(x)
		x = self.batchnorm1(x)
		x = self.maxpool1(x) 
		x = self.conv2(x)
		x = self.relu(x)
		x = self.batchnorm1(x)
		x = self.maxpool2(x)
		x = x.view(x.size(0), -1) 
		x = self.dropout(x)
		x = self.fc1(x) 
		x = self.leakyrelu(x)
		x = self.dropout(x)
		x = self.fc2(x)
		return x

## Create train loop functions

def train_step(model: torch.nn.Module,
               dataloader:torch.utils.data.DataLoader,
               loss_fn:torch.nn.Module,
               optimizer: torch.optim.Optimizer):
  model.train()
  train_acc,train_loss=0,0
  for batch, (X,y) in enumerate(dataloader):
    X,y= X.to(device),y.to(device)
    y_pred= model(X)
    loss = loss_fn(y_pred,y)
    train_loss+=loss
    y_pred_class= torch.argmax(torch.softmax(y_pred,dim=1), dim=1)
    train_acc+= accuracy_fn(y_pred_class,y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
  train_loss= train_loss/len(dataloader)
  train_acc= train_acc/len(dataloader)

  return train_loss,train_acc

## Create train loop functions

def test_step(model: torch.nn.Module,
               dataloader:torch.utils.data.DataLoader,
               loss_fn:torch.nn.Module):
  model.eval()
  test_acc,test_loss=0,0
  with torch.inference_mode():
    for batch, (X,y) in enumerate(dataloader):
      X,y= X.to(device),y.to(device)
      y_pred= model(X)
      loss = loss_fn(y_pred,y)
      test_loss+=loss
      y_pred_class= torch.argmax(torch.softmax(y_pred,dim=1),dim=1)
      test_acc+= accuracy_fn(y_pred_class,y)
    test_loss= test_loss/len(dataloader)
    test_acc= test_acc/len(dataloader)

  return test_loss,test_acc

def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module = nn.CrossEntropyLoss(),
          epochs: int=15):
  
  for epoch in tqdm(range(epochs)):
    train_loss,train_acc = train_step(model = model,
                                      dataloader= train_dataloader,
                                      loss_fn=loss_fn,
                                      optimizer=optimizer)

    test_loss,test_acc = test_step(model = model,
                                   dataloader= test_dataloader,
                                   loss_fn= loss_fn) 

    print(
        f"Epoch: {epoch+1} |"
        f"train_loss: {train_loss:.4f} |"
        f"train_acc: {train_acc:.4f} |"
        f"test_loss: {test_loss:.4f} |"
        f"test_acc: {test_acc: .4f}"
    )

"""**Train the Model**"""

torch.manual_seed(42)
torch.cuda.manual_seed(42)

NUM_EPOCHS = 15

model_Meso1= MesoNet_Attention1(input_features=3, output_features=2).to(device)
loss_fn= nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_Meso1.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-10)

train(model=model_Meso1,
                         train_dataloader=train_dataloader,
                         test_dataloader= test_dataloader,
                         optimizer=optimizer,
                         loss_fn= loss_fn,
                         epochs=NUM_EPOCHS)

"""**Predictions on the data**"""

def make_predictions(model: torch.nn.Module, data: list, device: torch.device = device):
    predict_proba = []
    model.eval()
    with torch.inference_mode():
        for sample in data:
            sample = torch.unsqueeze(sample, dim=0).to(device) # Add an extra dimension and send sample to device
            preds = model(sample)
            pred_prob = torch.softmax(preds.squeeze(), dim=0)
            predict_proba.append(pred_prob.cpu())
    return torch.stack(predict_proba)

import random
random.seed(52)
test_samples = []
test_labels = []
class_names= ["Fake","Real"]
for sample, label in random.sample(list(test_data), k=9):
    test_samples.append(sample)
    test_labels.append(label)

predict_proba= make_predictions(model=model_Meso1, 
                             data=test_samples)
pred_classes = predict_proba.argmax(dim=1)
pred_classes

plt.figure(figsize=(9,9))
rows=3
cols=3
torch.manual_seed(52)
for i,sample in enumerate(test_samples):
  plt.subplot(rows,cols,i+1)
  plt.imshow(sample.squeeze().permute(1,2,0),cmap="gray")
  pred_label= class_names[pred_classes[i]]
  truth_label= class_names[test_labels[i]]
  title_text= f"Predictions:{pred_label} | Truth: {truth_label}"
  if pred_label==truth_label:
    plt.title(title_text, fontsize=10, c="g")
  else:
    plt.title(title_text, fontsize=10, c='r')
  plt.axis(False)

# Pruning the Model
import torch.nn.utils.prune as prune

def measure_module_sparsity(module, weight=True, bias=False, use_mask=False):

    num_zeros = 0
    num_elements = 0

    if use_mask == True:
        for buffer_name, buffer in module.named_buffers():
            if "weight_mask" in buffer_name and weight == True:
                num_zeros += torch.sum(buffer == 0).item()
                num_elements += buffer.nelement()
            if "bias_mask" in buffer_name and bias == True:
                num_zeros += torch.sum(buffer == 0).item()
                num_elements += buffer.nelement()
    else:
        for param_name, param in module.named_parameters():
            if "weight" in param_name and weight == True:
                num_zeros += torch.sum(param == 0).item()
                num_elements += param.nelement()
            if "bias" in param_name and bias == True:
                num_zeros += torch.sum(param == 0).item()
                num_elements += param.nelement()

    sparsity = num_zeros / num_elements

    return num_zeros, num_elements, sparsity

def measure_global_sparsity(
    model, weight = True,
    bias = False, conv2d_use_mask = False,
    linear_use_mask = False):

    num_zeros = 0
    num_elements = 0

    for module_name, module in model.named_modules():

        if isinstance(module, torch.nn.Conv2d):

            module_num_zeros, module_num_elements, _ = measure_module_sparsity(
                module, weight=weight, bias=bias, use_mask=conv2d_use_mask)
            num_zeros += module_num_zeros
            num_elements += module_num_elements

        elif isinstance(module, torch.nn.Linear):

            module_num_zeros, module_num_elements, _ = measure_module_sparsity(
                module, weight=weight, bias=bias, use_mask=linear_use_mask)
            num_zeros += module_num_zeros
            num_elements += module_num_elements

    sparsity = num_zeros / num_elements

    return num_zeros, num_elements, sparsity

# To remove reparametrization

def remove_parameters(model):

    for module_name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            try:
                prune.remove(module, "weight")
            except:
                pass
            try:
                prune.remove(module, "bias")
            except:
                pass
        elif isinstance(module, torch.nn.Linear):
            try:
                prune.remove(module, "weight")
            except:
                pass
            try:
                prune.remove(module, "bias")
            except:
                pass

    return model

def pruning_func_unstruct(model, a, g=False):
    if g == True:
        # Global pruning
        parameters_to_prune = []
        for module_name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                parameters_to_prune.append((module, "weight"))

        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=a,
        )
        for module_name, module in model.named_modules():
          if isinstance(module, torch.nn.Conv2d):
              prune.remove(module, 'weight')
              pass
    else:
        for module_name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                prune.l1_unstructured(module,
                                      name="weight",
                                      amount=a)
                prune.remove(module, 'weight')
            elif isinstance(module, torch.nn.Linear):
                pass

    return model

def pruning_finetuning(model,
                      amount=0.4,
                      num_iterations=1,
                      num_epochs_per_iteration=2,
                      structured = False,
                      global_pruning=False):
  

  model.to(device)
  for i in range(num_iterations):

    print("Pruning and Finetuning {}/{}".format(i + 1, num_iterations))
    print("Pruning...")

    model = pruning_func_unstruct(model, amount, global_pruning)

    print("Fine-tuning...")

    train(model=model,
    train_dataloader=train_dataloader,
    test_dataloader= test_dataloader,
    optimizer=optimizer,
    loss_fn= loss_fn,
    epochs= num_epochs_per_iteration)

  return model

import copy

pruned_model = copy.deepcopy(model_Meso1)

print(measure_global_sparsity(
    pruned_model, weight = True,
    bias = False, conv2d_use_mask = False,
    linear_use_mask = False))

pruned_model = pruning_finetuning(pruned_model,
                      amount=0.2,
                      num_iterations=1,
                      num_epochs_per_iteration=5,
                      global_pruning=False)

pruned_model = remove_parameters(model=pruned_model)
