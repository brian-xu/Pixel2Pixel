import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
from multiprocessing.pool import Pool
from tqdm import tqdm
from PIL import Image
from sklearn.feature_extraction.image import extract_patches_2d
import os
import torch
import argparse
import torchvision.transforms as transforms
import time
from numba import jit
from numba import prange
parser = argparse.ArgumentParser('Pixel2Pixel')

parser.add_argument('--data-path', default='../data/test', type=str)
parser.add_argument('--dataset', default='SIDD/Noisy', type=str)
parser.add_argument('--save', default='./results', type=str)
parser.add_argument('--ws', default=40, type=int)
parser.add_argument('--ps', default=7, type=int)
parser.add_argument('--nn', default=100, type=int)
parser.add_argument('--loss', default='L2', type=str)

args = parser.parse_args()

torch.manual_seed(123)
torch.cuda.manual_seed(123)
np.random.seed(123)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

WINDOW_SIZE = args.ws
PATCH_SIZE = args.ps
NUM_NEIGHBORS = args.nn
loss_type = args.loss

transform = transforms.Compose([transforms.ToTensor()])

@jit(nopython=True)  # 设置nopython模式为True以获得最佳性能
def extract_patches_2d_np(image, patch_size):
    i_height, i_width, i_channel = image.shape[:3]
    p_height, p_width = patch_size

    # 计算补丁的数量
    n_patches_i = i_height - p_height + 1
    n_patches_j = i_width - p_width + 1

    # 初始化补丁数组
    patches = np.empty((n_patches_i * n_patches_j, p_height, p_width, i_channel), dtype=image.dtype)

    patch_idx = 0
    for i in range(n_patches_i):
        for j in range(n_patches_j):
            patches[patch_idx] = image[i:i + p_height, j:j + p_width]
            patch_idx += 1

    return patches

@jit(nopython=True)
def search_pixel(img, h, i, window_size, patch_size, num_neighbors, loss_type):
    x_ = np.zeros((h, num_neighbors, 3))
    for j in range(h):
        x = i + window_size // 2
        y = j + window_size // 2

        center_patch = img[x - (patch_size // 2): x + (patch_size // 2) + 1,
                           y - (patch_size // 2): y + (patch_size // 2) + 1]

        lx = max(0, x - window_size // 2)
        rx = min(img.shape[0] - patch_size, x + window_size // 2)
        ly = max(0, y - window_size // 2)
        ry = min(img.shape[1] - patch_size, y + window_size // 2)

        window = img[lx: rx + 1, ly: ry + 1]
        patches = extract_patches_2d_np(window, (patch_size, patch_size))

        # 计算与中心补丁的距离
        flat_patches = patches.reshape(patches.shape[0], -1)  # 展平补丁
        flat_center_patch = center_patch.flatten()  # 展平中心补丁
        if loss_type == 'L2':
            distances = np.sum((flat_patches - flat_center_patch) ** 2, axis=1)
        elif loss_type == 'L1':
            distances = np.sum(np.abs(flat_patches - flat_center_patch), axis=1)

        # 使用一维数组进行argsort
        sorted_indices = np.argsort(distances)
        best_indices = sorted_indices[:num_neighbors]

        # 从一维索引中获取最佳补丁
        best_patches = patches[best_indices]
        for n in range(num_neighbors):
            x_[j, n, :] = best_patches[n, patch_size // 2, patch_size // 2, :]

    return x_

@jit(nopython=True, parallel=True)
def process_image(img, w, h, window_size, patch_size, num_neighbors, loss_type):
    result = np.zeros((w, h, num_neighbors, 3))
    for i in prange(w):  # 使用 numba 的 prange 来并行循环
        result[i] = search_pixel(img, h, i, window_size, patch_size, num_neighbors, loss_type)
    return result


if __name__ == "__main__":
    root = os.path.join(args.save, '_'.join(str(i) for i in [args.dataset, args.ws, args.ps, args.nn, args.loss]))

    os.makedirs(root, exist_ok=True)
    image_folder = os.path.join(args.data_path, args.dataset)
    image_files = [f for f in os.listdir(image_folder)]
    image_files = sorted(image_files)
    for image_file in image_files:
        start_time = time.time()
        image_path = os.path.join(image_folder, image_file)

        img = Image.open(image_path)
        img = transform(img)
        img = img.numpy().transpose((1, 2, 0))
        end1 = time.time()
        print("transform time:", end1 - start_time)

        w, h = img.shape[0], img.shape[1]
        img = np.pad(img, ((WINDOW_SIZE//2, WINDOW_SIZE//2), (WINDOW_SIZE//2, WINDOW_SIZE//2), (0, 0)), 'symmetric')
        X = process_image(img, w, h, WINDOW_SIZE, PATCH_SIZE, NUM_NEIGHBORS, loss_type)

        end2 = time.time()
        print("compute time:", end2-end1)

        file_name_without_extension = os.path.splitext(image_file)[0]
        np.save(os.path.join(root, file_name_without_extension), X) # HxWxNxC
        print(f"Processing {image_file}")
        end_time = time.time()

        total_time = end_time - start_time
        print("Total execution time: {:.2f} seconds".format(total_time))
    print('All subprocesses done.')