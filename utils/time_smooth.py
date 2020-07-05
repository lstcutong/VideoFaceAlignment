import torch
import torch.nn as nn
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
import copy

SMOOTH_METHODS_OPTIMIZE = 1
SMOOTH_METHODS_MEDUIM = 2
SMOOTH_METHODS_MEAN = 3
SMOOTH_METHODS_GAUSSIAN = 4
SMOOTH_METHODS_DCNN = 5


# from networks import TimeSmoothNetWork
class Model(nn.Module):
    def __init__(self, H, W, initation=None):
        super(Model, self).__init__()

        self.H = H
        self.W = W
        self.l1 = nn.Linear(self.W, self.H)

        if initation is not None:
            self.l1.weight.data = torch.from_numpy(initation)

        self.smo_param = self.l1.weight

    def divergence(self, x):
        g_x = x[:, 1:self.W] - x[:, 0:self.W - 1]

        g_xx = g_x[:, 1:self.W - 1] - g_x[:, 0:self.W - 2]

        return g_xx

    def gradient(self, x):
        g_x = x[:, 1:self.W] - x[:, 0:self.W - 1]
        return g_x

    def forward(self, ref_param):
        sim_loss = torch.sum((ref_param - self.smo_param.float()) ** 2)

        smo_loss = torch.norm(self.divergence(self.smo_param.float()), p=2)
        smo_loss2 = torch.norm(self.gradient(self.smo_param.float()), p=2)
        return sim_loss, smo_loss


'''
提供5种平滑方式
均值平滑，中值平滑，高斯平滑，，基于优化的平滑，卷积网络平滑
输入: ref_param 参考参数，类型 ndarray, shape:[param_num,frames]
输出: new_param 平滑参数, 类型 ndarray, shape:[param_num,frames]
'''


def smooth_optimize(ref_param):
    ref_param = ref_param.astype(np.float32)
    H, W = ref_param.shape

    model = Model(H, W, initation=ref_param).cuda()
    ref_param = torch.from_numpy(ref_param).float().cuda()

    iterations = 300
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # print(ref_param.shape)

    for it in range(iterations):
        sim_loss, smo_loss = model(ref_param)

        loss = 1 * sim_loss + 1.3 * smo_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    new_param = model.smo_param.cpu().detach().numpy()

    return new_param


def smooth_medium_filter(ref_param, k=5):
    assert k % 2 == 1, "未实现偶数步长"
    H, W = ref_param.shape

    s = int(k / 2)

    new_param = copy.deepcopy(ref_param)
    for i in range(0, W):
        start = np.maximum(0, i - s)
        end = np.minimum(W, i + s + 1)

        new_param[:, i] = np.median(ref_param[:, start:end], axis=1)

    return new_param


def smooth_mean_filter(ref_param, k=5):
    assert k % 2 == 1, "未实现偶数步长"
    H, W = ref_param.shape

    s = int(k / 2)

    new_param = copy.deepcopy(ref_param)
    for i in range(0, W):
        start = np.maximum(0, i - s)
        end = np.minimum(W, i + s + 1)

        new_param[:, i] = np.mean(ref_param[:, start:end], axis=1)

    return new_param


def smooth_gaussian_filter(ref_param, k=5, miu=0, sigma=1):
    assert k % 2 == 1, "未实现偶数步长"
    H, W = ref_param.shape

    center = int(k / 2)
    x = np.array([i - center for i in range(k)])

    weights = np.exp(-(x - miu) ** 2 / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))
    weights = weights / np.sum(weights)

    new_param = copy.deepcopy(ref_param)

    for i in range(center, W - center):
        start = np.maximum(0, i - center)
        end = np.minimum(W, i + center + 1)

        new_param[:, i] = ref_param[:, start:end] @ weights

    return new_param


def smooth_DCNN(ref_param):
    pass


def calculate_smooth_loss(x):
    H, W = x.shape

    g_x = x[:, 1:W] - x[:, 0:W - 1]

    g_xx = g_x[:, 1:W - 1] - g_x[:, 0:W - 2]

    return np.sum(g_xx ** 2)
    # return np.sum(np.abs(g_xx))


def calculate_sim_loss(x, y):
    return np.sum((x - y) ** 2)


def start_smooth_param(ref_param, method=SMOOTH_METHODS_GAUSSIAN, win_size=5, miu=0, sigma=1):
    if method == SMOOTH_METHODS_DCNN:
        return smooth_DCNN(ref_param)

    if method == SMOOTH_METHODS_GAUSSIAN:
        return smooth_gaussian_filter(ref_param, miu=miu, sigma=sigma, k=win_size)

    if method == SMOOTH_METHODS_MEAN:
        return smooth_mean_filter(ref_param, k=win_size)

    if method == SMOOTH_METHODS_MEDUIM:
        return smooth_medium_filter(ref_param, k=win_size)

    if method == SMOOTH_METHODS_OPTIMIZE:
        return smooth_optimize(ref_param)


'''
def testTimeSmoothNet(ref_param):
    H,W = ref_param.shape
    model = TimeSmoothNetWork().cuda()
    ref_param = torch.from_numpy(ref_param).float().cuda()
    ref_param = torch.stack([ref_param])
    ref_param = torch.stack([ref_param])

    iterations = 800000
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-4)

    def adjust_learning_rate(optimizer):
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] / 2

    def lossfunc(x,y):
        sim_loss1 = torch.sum((x-y)**2)

        g_x = x[:,:,:,1:W] - x[:,:,:,0:W-1]
        g_xx = g_x[:,:,:,1:W-1] - g_x[:,:,:,0:W-2]

        smo_loss1 = torch.norm(g_xx,p=2) + torch.norm(g_x,p=2)
        return sim_loss1 ,smo_loss1

    for it in range(iterations):
        output = model(ref_param)
        sim_loss, smo_loss = lossfunc(output,ref_param)
        loss =  sim_loss +  10 * smo_loss

        if (it+1) % 10000 == 0:
            adjust_learning_rate(optimizer)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print("{}/{} sim_loss:{:.6f} smo_loss:{:.6f}".format(it+1,iterations,sim_loss,smo_loss))

    new_param = model(ref_param)[0][0].cpu().detach().numpy()

    #print(new_param.shape)
    return new_param

'''
