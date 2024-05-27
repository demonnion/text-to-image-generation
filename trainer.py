# !pip install scikit-learn

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import random
import numpy as np
import matplotlib.pyplot as plt
# import wandb
import os
import pickle
from torch import nn
# from tqdm.notebook import tqdm
from tqdm import tqdm
import json
from PIL import Image
from torchvision import transforms, utils
from torchvision import models
import torch.optim as optim
from copy import deepcopy
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import plotly.express as px

root = "."

device = "cuda:1" if torch.cuda.is_available() else "cpu"

# Ensure deterministic behavior taken from wandb docs
torch.backends.cudnn.deterministic = True
random.seed(3407)
np.random.seed(3407)
torch.manual_seed(3407)
torch.cuda.manual_seed_all(3407)

# !wget https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz

# !wget https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat

# !wget https://www.robots.ox.ac.uk/~vgg/data/flowers/102/setid.mat

import scipy.io

# !tar -xvf 102flowers.tgz

# !tar -xvf CUB_200_2011.tgz

image_labels_mat = scipy.io.loadmat('imagelabels.mat')

print(len(image_labels_mat['labels'][0]))
print(image_labels_mat['labels'][0][0])

image_cat_labels = np.array(image_labels_mat['labels'][0])

print(image_cat_labels.min())
print(image_cat_labels.max())
print(image_cat_labels.shape)

set_id_mat = scipy.io.loadmat('setid.mat')

print(len(set_id_mat["trnid"][0]))
print(len(set_id_mat["valid"][0]))
print(len(set_id_mat["tstid"][0]))

"""Load the Images"""

# IMG_DIR = '/content/jpg'
IMG_DIR = './jpg'
IMG_SIZE = (128,128)
LABEL_PATH = 'imagelabels.mat'

class Dataset(torch.utils.data.Dataset):
    def init(self, base_path, labels_path, transform=None):
        image_labels_mat = scipy.io.loadmat(labels_path)
        self.labels = np.array(image_labels_mat['labels'][0])
        print(self.labels)
        img_dir_list = sorted(os.listdir(base_path))
        self.image_paths = [os.path.join(base_path,img_path) for img_path in img_dir_list]
        self.transform = transform

    def len(self):
        return len(self.labels)

    def getitem(self, idx):
        image = np.array(Image.open(self.image_paths[idx]))
        label = self.labels[idx]
        # print(label)
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)

    def get_label(self, idx):
        return self.labels[idx]

    # def get_image(idx):
    #     image = np.array(Image.open(self.image_paths[idx]))
    #     return image

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(IMG_SIZE)
])

train_data = Dataset(IMG_DIR,LABEL_PATH, transform)

train_data[-1][1]

def visualise_Class(class_num):
    train_data = Dataset(IMG_DIR,LABEL_PATH, transform)
    # images = []
    num_found = 0
    for i in range(8000):
        if train_data.get_label(i) == class_num:
            # images.append(train_data[i][0])
            num_found += 1
            plt.imshow(train_data[i][0].permute(1,2,0))
            plt.show()


        if num_found > 10:
            break

visualise_Class(55)

class LabelEmbeds(nn.Module):
    def init(self, embed_len):
        super(LabelEmbeds,self).init()
        self.embed_out = embed_len
        self.embed_layer = nn.Sequential(
            nn.Linear(in_features=1, out_features = embed_len),
            nn.ReLU()
        )

    def forward(self, labels):
        return self.embed_layer(labels)

"""Generator"""

class GeneratorNetwork(nn.Module):
    # stride=4 implies that every layer will increase the Height and Width by a factor of 2
    
    def init(self, embed_network, len_input, layer_channels):
        super(GeneratorNetwork,self).init()

        self.len_input = len_input

        self.embed_network = embed_network

        self.class_embed_layer = nn.Sequential(
            nn.Linear(in_features=self.embed_network.embed_out, out_features = len_input//5),
            nn.ReLU()
        )

        self.gen_layer_1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=len_input, out_channels=layer_channels*8, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(layer_channels*8),
            nn.ReLU(inplace=True)
        )

        self.gen_layer_2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=layer_channels*8, out_channels=layer_channels*4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(layer_channels*4),
            nn.ReLU(inplace=True)
        )

        self.gen_layer_3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=layer_channels*4, out_channels=layer_channels*2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(layer_channels*2),
            nn.ReLU(inplace=True)
        )

        self.gen_layer_4 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=layer_channels*2, out_channels=layer_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(layer_channels),
            nn.ReLU(inplace=True)
        )

        self.gen_layer_5 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=layer_channels, out_channels=3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )
        
        self.gan_all_layers = nn.Sequential(
            self.gen_layer_1,
            self.gen_layer_2,
            self.gen_layer_3,
            self.gen_layer_4,
            self.gen_layer_5
        )


    def forward(self, class_labels, input_vec=None):
        if input_vec == None:
            # sample from gaussian dist
            input_vec = torch.randn(class_labels.shape[0],4 * (self.len_input//5), 1, 1).to(device)
    
        embeds = self.embed_network(class_labels)
        condition_input = self.class_embed_layer(embeds).reshape(class_labels.shape[0], -1, 1, 1)

        gan_input = torch.cat([input_vec, condition_input], dim=1)

        gan_out = self.gan_all_layers(gan_input)
        return gan_out

"""Discrimintor"""

# b c h w
# add new channel 
class DiscriminatorNetwork(nn.Module):
    def init(self, embed_network, len_input, layer_channels):
        super(DiscriminatorNetwork, self).init()
        self.len_input = len_input

        self.embed_network = embed_network

        self.class_embed_layer = nn.Sequential(
            nn.Linear(in_features=self.embed_network.embed_out, out_features = self.len_input),
            nn.ReLU()
        )

        # input: 4 x h x w, out: lc x h/2 x w/2
        self.des_layer_1 = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=layer_channels, kernel_size=4, stride=2, padding=1, bias=True),
            nn.LeakyReLU(0.2)
        )

        # input: lc x h/2 x w/2, out: lc * 2 x h/4 x w/4
        self.des_layer_2 = nn.Sequential(
            nn.Conv2d(in_channels=layer_channels, out_channels=layer_channels * 2, kernel_size=4, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(num_features=layer_channels * 2),
            nn.LeakyReLU(negative_slope=0.2)
        )

        # input: lc * 2 x h/4 x w/4, out: lc * 4 x h/8 x w/8
        self.des_layer_3 = nn.Sequential(
            nn.Conv2d(in_channels=layer_channels * 2, out_channels=layer_channels * 4, kernel_size=4, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(num_features=layer_channels * 4),
            nn.LeakyReLU(negative_slope=0.2)
        )

        # input: lc * 4 x h/8 x w/8, out: lc * 8 x h/16 x w/16
        self.des_layer_4 = nn.Sequential(
            nn.Conv2d(in_channels=layer_channels * 4, out_channels=layer_channels * 8, kernel_size=4, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(num_features=layer_channels * 8),
            nn.LeakyReLU(negative_slope=0.2)
        )

        self.classification_head = nn.Sequential(
            nn.LazyLinear(out_features=1),
            nn.Sigmoid()
        )

        self.des_conv_layers = nn.Sequential(
            self.des_layer_1,
            self.des_layer_2,
            self.des_layer_3,
            self.des_layer_4,
        )
    
    def forward(self, images, class_labels):
        # b, L
        embeds = self.embed_network(class_labels)
        # b, L, 1
        embeds = self.class_embed_layer(embeds).reshape(class_labels.shape[0], self.len_input, 1)
        # b, 1, L, L 
        embeds = embeds.repeat(1,1,self.len_input).reshape(class_labels.shape[0], 1, self.len_input, self.len_input)
        input = torch.cat([images, embeds], dim=1)
        out_conv = self.des_conv_layers(input).reshape(class_labels.shape[0],-1)
        return self.classification_head(out_conv)

Config = {
    'batch_size':128,
    'device': device,
    'input_len': 100,
    'dis_input_len':64,
    'learning_rate': 2e-4,
    'epochs': 302,
    'image_size': (64, 64),
    'last_epoch': 0
}

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(Config["image_size"]),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


def gen_image(gen_model, config, epoch, class_nums=None, noise=None):
    gen_model.eval()
    if class_nums == None:
        class_nums = torch.Tensor([i+1 for i in range(102)]).reshape(102,1).to(device)
        noise = torch.randn(102, 4* (config["input_len"]//5), 1, 1).to(device)
    
    gen_images = gen_model(class_nums.float(), noise)
    gen_images = (gen_images + 1) * 127.5

    fig, axs = plt.subplots(17, 6, figsize=(20, 40))
    axs = axs.flatten()

    for i in range(102):
        axs[i].imshow(gen_images[i].permute(1,2,0).long().cpu().numpy(), vmin=0, vmax=255)
        axs[i].set_title(i)
        
    plt.tight_layout()
    plt.savefig(os.path.join(root, f"images_2/{epoch}.png"))
    # plt.show()

    gen_model.train()
    return gen_images




def train(gen_model, dis_model, config):
    gen_model.train()
    dis_model.train()
    
    gen_optimizer = torch.optim.Adam(gen_model.parameters(), lr=config["learning_rate"], betas=(0.5, 0.999))
    dis_optimizer = torch.optim.Adam(dis_model.parameters(), lr=config["learning_rate"], betas=(0.5, 0.999))
    criterion = nn.BCELoss()
    
    train_loader = DataLoader(Dataset(IMG_DIR,LABEL_PATH, transform), batch_size=config["batch_size"], shuffle=True)

    val_noise = torch.randn(102, 4* (config["input_len"]//5), 1, 1).to(device)
    val_labels = torch.Tensor([i+1 for i in range(102)]).reshape(102,1).to(device)


    gen_losses = []
    dis_losses = []
    gen_images = [] 
    dis_guess = []

    for epoch in tqdm(range(config["last_epoch"], config["last_epoch"] + config["epochs"])):
        running_dis_loss = 0.0
        running_gen_loss = 0.0

        for batch, (images, labels) in enumerate(tqdm(train_loader)):
            images = images.to(device)
            labels = labels.to(device).reshape(-1, 1)

            # Train Discriminator 
            dis_model.zero_grad()

            dis_out = dis_model(images, labels.float())

            loss = criterion(dis_out.reshape(-1), torch.ones(labels.shape[0]).to(device))
            loss.backward()
            dis_loss = loss.item()
            dis_guess.append(dis_out.argmax(dim=1).sum().item()/images.shape[0])
            fake_images = gen_model(labels.float())
            fake_images = fake_images.detach()
            dis_out = dis_model((fake_images + 1.0)/2, labels.float())

            loss = criterion(dis_out.reshape(-1), torch.zeros(labels.shape[0]).to(device))
            loss.backward()
            dis_loss += loss.item()
            dis_optimizer.step()

            # Train Generator
            gen_model.zero_grad()
            fake_images = gen_model(labels.float())
            dis_out = dis_model((fake_images + 1.0)/2, labels.float())
            loss = criterion(dis_out.reshape(-1), torch.ones(labels.shape[0]).to(device))

            loss.backward()
            gen_optimizer.step()

            gen_loss = loss.item()

            gen_losses.append(gen_loss)
            dis_losses.append(dis_loss)

            if batch % 32 == 0:
            #     gen_model.eval()
            #     fake_images = (gen_model(val_labels.float(), val_noise) + 1) * 127.5
            #     # plt.imshow(fake_images[0].permute(1,2,0).long().cpu().numpy(), vmin=0, vmax=255)
            #     # plt.show()
            #     fake_images = gen_image(gen_model, config, epoch+1, val_labels.float(), val_noise)
            #     gen_images.append(fake_images.long().cpu().numpy())
            #     gen_model.train()

                print(f"Epoch: {epoch+1}, batch: {batch+1}, Generator Loss: {gen_loss}, Discriminator Loss: {dis_loss}, Discriminator Guess: {dis_guess[-1]}")

            
        fake_images = gen_image(gen_model, config, epoch+1, val_labels.float(), val_noise)
        gen_images.append(fake_images.long().cpu().numpy())


        # save model 
        if (epoch+1) % 10 == 0:
            print("Saving checkpoint..", os.path.join(root, f'final_2/{epoch}_dis.pth'))
            torch.save(gen_model.state_dict(), os.path.join(root, f'final_2/{epoch}_gen.pth'))
            torch.save(dis_model.state_dict(), os.path.join(root, f'final_2/{epoch}_dis.pth'))

    return gen_losses, dis_losses, gen_images, val_labels, dis_guess

def make(config):
    embed_network = LabelEmbeds(config["input_len"]//2).to(config["device"])
    gen_model = GeneratorNetwork(embed_network, len_input=100, layer_channels=96).to(config["device"])
    dis_model = DiscriminatorNetwork(embed_network, len_input=config["dis_input_len"], layer_channels=64).to(config["device"])

    return gen_model, dis_model

gen_model, dis_model = make(Config)

print(Config["device"])

gen_losses, dis_losses, gen_images, val_labels, dis_guess= train(gen_model, dis_model, Config)


data_save = {
    "gen_losses": gen_losses,
    "dis_losses": dis_losses,
    "gen_images": gen_images,
    "val_labels": val_labels,
    "dis_guess": dis_guess
}

with open("./pickles_2/data_sace.pickle", "wb") as f:
    pickle.dump(data_save, f)




# !mkdir /content/drive/MyDrive/DL-CSE641/final

# gen_model, dis_model = make(Config)

# gen_model.load_state_dict(torch.load(os.path.join(root, f'final/{109}_gen.pth')))
# dis_model.load_state_dict(torch.load(os.path.join(root, f'final/{109}_dis.pth')))

# gen_image(gen_model, Config)
