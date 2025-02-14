import numpy as np
import matplotlib.pyplot as plt
# from tqdm.notebook import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.init as init
# Load test image from URL
import torchvision.transforms as transforms
import requests
from io import BytesIO
import os
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from torch.optim.lr_scheduler import StepLR, MultiStepLR
from torchvision.transforms.functional import to_pil_image
from skimage import io



import argparse
import time


torch.manual_seed(123)
torch.cuda.manual_seed(123)
np.random.seed(123)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# 4 test images from the Kodak24 dataset
device = "cuda:0"
#
# response = requests.get(url)
# path=BytesIO(response.content)
#
# clean_img = torch.load(path).unsqueeze(0)
parser = argparse.ArgumentParser('Pixel2Pixel')
parser.add_argument('--data-path', default='../data/test', type=str)
parser.add_argument('--dataset', default='SIDD140', type=str)
parser.add_argument('--gt', default='GT', type=str)
parser.add_argument('--save', default='./results', type=str)
parser.add_argument('--out_image', default='./results_image', type=str)
parser.add_argument('--ws', default=40, type=int)
parser.add_argument('--ps', default=7, type=int)
parser.add_argument('--nn', default=100, type=int)
parser.add_argument('--mm', default=20, type=int)
parser.add_argument('--loss', default='L2', type=str)

args = parser.parse_args()


image_folder = os.path.join(args.data_path, args.dataset, args.gt)
image_files = [f for f in os.listdir(image_folder) ]
image_files = sorted(image_files)

sim_image_folder = os.path.join(args.save, args.dataset, '_'.join(str(i) for i in [args.ws, args.ps, args.nn, args.loss]))
sim_image_files = [f for f in os.listdir(sim_image_folder)]
sim_image_files = sorted(sim_image_files)

transform = transforms.Compose([transforms.ToTensor()])

max_epoch = 3000
lr = 0.0001
# step_size = 1500
# gamma = 0.5


class network(nn.Module):
    def __init__(self, n_chan, chan_embed=64):
        super(network, self).__init__()

        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv1 = nn.Conv2d(n_chan, chan_embed, 3, padding=1)
        self.conv2 = nn.Conv2d(chan_embed, chan_embed, 3, padding=1)
        self.conv4 = nn.Conv2d(chan_embed, chan_embed, 3, padding=1)
        self.conv5 = nn.Conv2d(chan_embed, chan_embed, 3, padding=1)
        self.conv6 = nn.Conv2d(chan_embed, chan_embed, 3, padding=1)
        # self.conv7 = nn.Conv2d(chan_embed, chan_embed, 3, padding=1)
        # self.conv8 = nn.Conv2d(chan_embed, chan_embed, 3, padding=1)
        # self.conv9 = nn.Conv2d(chan_embed, chan_embed, 3, padding=1)
        self.conv3 = nn.Conv2d(chan_embed, n_chan, 1)

        self._initialize_weights()

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.act(self.conv4(x))
        x = self.act(self.conv5(x))
        x = self.act(self.conv6(x))
        # x = self.act(self.conv7(x))
        # x = self.act(self.conv8(x))
        # x = self.act(self.conv9(x))
        x = self.conv3(x)
        return x
        # return torch.sigmoid(x)


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.orthogonal_(m.weight)
                # print('init weight')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)


def mse(gt: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    loss = torch.nn.MSELoss()
    return loss(gt, pred)


loss_f = nn.L1Loss() if args.loss == 'L1' else nn.MSELoss()


def loss_func(img1, img2, loss_f=nn.MSELoss()):
    # pred1 = model(img1)
    # pred2 = model(img2)
    #
    # loss = 1 / 2 * (loss_f(img1, pred2) + loss_f(img2, pred1))
    pred1 = model(img1)
    # pred2 = model(img2)

    loss = loss_f(img2, pred1)
    return loss



def train(model, optimizer, img_bank):
    # prepare a noise2noise pair
    index1 = torch.randint(0, N, size=(H * W, 1)).to(device)
    img1 = torch.gather(img_bank, 0, index=index1.expand_as(img_bank))[0]
    img1 = img1.view(1, H, W, C).permute(0, 3, 1, 2)


    index2 = torch.randint(0, N, size=(H * W, 1)).to(device)
    index2[index2==index1] = (index2[index2==index1] + 1) % N
    img2 = torch.gather(img_bank, 0, index=index2.expand_as(img_bank))[0]
    img2 = img2.view(1, H, W, C).permute(0, 3, 1, 2)

    loss = loss_func(img1, img2, loss_f)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()

# def train(model, optimizer, img_bank):
#     # prepare a noise2noise pair
#     # index1 = torch.randint(0, N, size=(1,)).to(device)
#     img1 = img_bank[0]
#     img1 = img1.view(1, H, W, C).permute(0, 3, 1, 2)
#
#
#     # index2 = torch.randint(0, N, size=(1,)).to(device)
#     # index2[index2==index1] = (index2[index2==index1] + 1) % N
#     img2 = img_bank[1]
#     img2 = img2.view(1, H, W, C).permute(0, 3, 1, 2)
#
#     loss = loss_func(img1, img2, loss_f)
#
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
#
#     return loss.item()


def test(model, noisy_img, clean_img):
    with torch.no_grad():
        pred = torch.clamp(model(noisy_img), 0, 1)
        MSE = mse(clean_img, pred).item()
        PSNR = 10 * np.log10(1 / MSE)

    return PSNR, pred

avg_PSNR = 0
avg_SSIM = 0
root = os.path.join(args.out_image, '_'.join(
        str(i) for i in [args.dataset, args.loss]))
os.makedirs(root, exist_ok=True)
for image_file in image_files:
    # start_time = time.time()
    image_path = os.path.join(image_folder, image_file)
    clean_img = Image.open(image_path)
    # Convert image to tensor and add an extra batch dimension
    clean_img = transform(clean_img).unsqueeze(0)

    clean_img_1 = io.imread(image_path)

    sim_img_path = os.path.join(sim_image_folder,
                                str(image_file).replace("png", "npy").replace("jpg", "npy").replace("PNG","npy").replace("JPG", "npy").replace("tif", "npy"))

    img_bank = np.load(sim_img_path).astype(np.float32).transpose((2, 0, 1, 3)) # NxHxWxC


    noisy_img = torch.from_numpy(img_bank[:1].transpose(0, 3, 1, 2))

    img_bank = img_bank[:args.mm]

    N, H, W, C = img_bank.shape
    img_bank = torch.from_numpy(img_bank).view(img_bank.shape[0], -1, img_bank.shape[-1]).to(device)



    clean_img = clean_img.to(device)
    noisy_img = noisy_img.to(device)

    n_chan = clean_img.shape[1]
    model = network(n_chan)

    model = model.to(device)
    print("The number of parameters of the network is: ", sum(p.numel() for p in model.parameters() if p.requires_grad))


    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = MultiStepLR(optimizer, milestones=[1500,2000,2500], gamma=0.5)


    for epoch in range(max_epoch):
        train(model, optimizer, img_bank)
        scheduler.step()

        # if (epoch + 1) % 100 == 0:
    PSNR, out_img = test(model, noisy_img, clean_img)



    out_img = to_pil_image(out_img.squeeze(0))
    out_img.save(os.path.join(root, os.path.splitext(image_file)[0] + '.png'))
    # end_time = time.time()
    noisy_img = to_pil_image(noisy_img.squeeze(0))
    noisy_img.save(os.path.join(root, os.path.splitext(image_file)[0] + '.PNG'))


    out_img = io.imread(os.path.join(root, os.path.splitext(image_file)[0] + '.png'))

    SSIM, _ = compare_ssim(clean_img_1, out_img, full=True, multichannel=True)
    print(f"PSNR for {image_file}: {PSNR:.2f}, SSIM: {SSIM:.4f}")

    avg_PSNR += PSNR
    avg_SSIM += SSIM

    # 计算并打印总运行时间
    # total_time = end_time - start_time
    # print("Total execution time: {:.2f} seconds".format(total_time))

avg_PSNR = avg_PSNR / len(image_files)
avg_SSIM = avg_SSIM / len(image_files)
print(f"PSNR for avg_PSNR: {avg_PSNR}, SSIM: {avg_SSIM}")
