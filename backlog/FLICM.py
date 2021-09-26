import random

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


# 初始化
def init_center(data):
    x = random.randint(0,data.shape[0])
    y = random.randint(0,data.shape[1])
    tmp = data[x][y]
    return tmp
def produce_center(K, data_size, data):
    a = np.ones((K,data_size))
    for k in range(K):
        center = init_center(data)
        a[k] = center
    return a
def init_weight(K):
# 产生单个像素点对于K个聚类中心的隶属度
# 格式为 u = [] 其中该元素为属于第一类的隶属度，以此类推
    u = []
    for i in range(K):
        rand = random.random()
        u.append(rand)
    sum_u = sum(u)
    for i in range(len(u)):
        u[i] = u[i]/sum_u
    return u
def produce_weight(data,K):
    row = data.shape[0]
    column = data.shape[1]
    weight = np.zeros((row, column, K))
    for i in range(row):
        for j in range(column):
            miu = init_weight(K)
            for k in range(K):
                weight[i][j][k] = miu[k]
    return weight
def neighbor_distance(data):
    row = data.shape[0]
    column = data.shape[1]
    dx = [-1, 0, 1, -1, 1, -1, 0, 1]
    dy = [-1, -1, -1, 0, 0, 1, 1, 1]
    neighbor_dist = np.zeros((row, column, 8))
    for i in range(1, row-1):
        for j in range(1, column-1):
                for r in range(8):
                    index_x = i + dx[r]
                    index_y = j + dy[r]
                    neighbor_dist[i][j][r] = pow( (data[index_x][index_y]-data[i][j]) ,2)
    return neighbor_dist
# 模糊因子
def Gki(data, K, center, weight, m = 2):
    row = data.shape[0]
    column = data.shape[1]
    dx = [-1, 0, 1, -1, 1, -1, 0, 1]
    dy = [-1, -1, -1, 0, 0, 1, 1, 1]
    d = [1.414, 1, 1414, 1, 1, 1.414, 1, 1.414]
    g = np.zeros((row, column, K))
    neighbor_dist = np.ones((row, column, 8))
    for i in range(1, row-1):
        # print("i Gki:", i)
        for j in range(1, column-1):
            for k in range(K):
                for r in range(8):
                    index_x = i + dx[r]
                    index_y = j + dy[r]
                    # neighbor_dist[i][j][r] = pow((data[index_x][index_y]-data[i][j]),2)
                    g[i][j][k] += (1/(d[r]+1)) * pow((1 - weight[index_x][index_y][k]), m) * distance(data[index_x][index_y], center[k])

    return g

def distance(a, b):
    return pow(np.linalg.norm(a - b), 2)
def cal_weight(data, center, gki, K, m = 2 ):
    row = data.shape[0]
    column = data.shape[1]
    dist= np.zeros((row, column, K))
    # 计算像素和聚类中心之间的距离
    for i in range(row):
        # print("i(weight):", i)
        for j in range(column):
            for k in range(K):
                dist[i][j][k] = distance(data[i][j], center[k]) + 0.001

    miu_factors = np.zeros((row, column, K))
    miu = np.zeros((row, column, K))
    for i in range(row):
        for j in range(column):
            for k in range(K):
                miu_factors[i][j][k] = gki[i][j][k] + dist[i][j][k]
    for i in range(row):
        for j in range(column):
            for k in range(K):
                for l in range(K):
                    miu[i][j][k] += pow( miu_factors[i][j][k] / miu_factors[i][j][l] , 1/(m -1))
                miu[i][j][k] = 1/miu[i][j][k]

    return miu
def cal_center(weight, data, K, m = 2):
    row = data.shape[0]
    column = data.shape[1]
    center_mu = np.zeros((K, 1))
    center_z = np.zeros((K, 3))
    for k in range(K):
        for i in range(row):
            # print("i center:",i)
            for j in range(column):
                center_mu[k] += pow(weight[i][j][k], m)
                center_z[k] += pow(weight[i][j][k], m) * data[i][j]
    c = np.zeros((K, 3))
    for k in range(K):
        c[k] = center_z[k]/center_mu[k]
    return c
def separation(data, weight,center):
    index = 0
    for i in range(weight.shape[0]):
        for j in range(weight.shape[1]):
            tmp = 0
            for k in range(weight.shape[2]):
                if weight[i][j][k]>=tmp:
                    tmp = weight[i][j][k]
                    index = k
            data[i][j] = center[index][0]
    return data
# 基于熵的评价
def e_value(weight,data2):
    index = 0
    hr = hl = 0
    K = weight.shape[2]
    cnt = np.zeros((K, 1))#记录每类的像素点数目
    label = np.zeros(data2.shape)#每个像素点的类别（根据隶属度确定）
    h = np.zeros((1, K))#区域熵
    LS = np.zeros((weight.shape[2], 256))#每一类里 像素等于某个灰度值的像素点数目
    row, column = data2.shape
    for i in range(weight.shape[0]):
        for j in range(weight.shape[1]):
            tmp = 0
            for k in range(K):
                if weight[i][j][k] >= tmp:
                    tmp = weight[i][j][k]
                    index = k
            cnt[index][0] += 1
            label[i][j] = index
    for i in range(row):
        for j in range(column):
            x = int(label[i][j])
            y = int(data2[i][j])
            LS[x][y] += 1
    s = np.sum(LS, axis=1)

    for i in range(LS.shape[0]):
        for j in range(LS.shape[1]):
            if LS[i][j] != 0:
                h[0][i] += -LS[i][j]/(s[i]+0.1)*np.log(LS[i][j]/(s[i]+0.1))
    for k in range(K):
        hr += cnt[k][0]/np.sum(cnt)*h[0][k]
        hl += -(cnt[k]/np.sum(cnt))*np.log((cnt[k]/np.sum(cnt)))
    E = hr + hl
    return E
# 区域间方差评价标准
def DIR(data, weight):
    index = 0
    K = weight.shape[2]
    label = np.zeros(data.shape)
    re_data = [[] for _ in range(K)]
    mean_data = Nk = [0] * K
    row, column = data.shape
    for i in range(row):
        for j in range(column):
            tmp = 0
            for k in range(K):
                if weight[i][j][k] >= tmp:
                    tmp = weight[i][j][k]
                    index = k
                label[i][j] = index

    for i in range(row):
        for j in range(column):
                re_data[int(label[i][j])].append(data[i][j])
    for k in range(K):
        mean_data[k] = sum(re_data[k])/len(re_data[k])
    mx = np.max(data)
    mn = np.min(data)
    return abs(mean_data[0]-mean_data[1])/(mx - mn)
#合成图 有监督正确率计算
def cal_correct(tlabel, weight):
    row,column,K = weight.shape
    # label = np.zeros([row, column])
    label = np.zeros(tlabel.shape)
    cnt = 0
    for i in range(row):
        for j in range(column):
            tmp = 0
            index = 0
            for k in range(K):
                if weight[i][j][k]>=tmp:
                    tmp = weight[i][j][k]
                    index = k
            label[i][j] = index
            if label[i][j] == tlabel[i][j]:
                cnt += 1
    if cnt/(row*column)<=0.5:
        return 1-cnt/(row*column)
    else:
        return cnt/(row*column)

def confusion_matrix(tlabel, weight):
    row,column,K = weight.shape
    # label = np.zeros([row, column])
    plabel = np.zeros(tlabel.shape)
    tp = fn = fp = tn = 0
    for i in range(row):
        for j in range(column):
            tmp = 0
            index = 0
            for k in range(K):
                if weight[i][j][k]>=tmp:
                    tmp = weight[i][j][k]
                    index = k
            plabel[i][j] = index
            if tlabel[i][j] == 0 and plabel[i][j] == 0:
                tp += 1
            elif tlabel[i][j] == 0 and plabel[i][j] == 1:
                fn += 1
            elif tlabel[i][j] == 1 and plabel[i][j] == 0:
                fp += 1
            elif tlabel[i][j] == 1 and plabel[i][j] == 1:
                tn += 1
    return [[tp,fn],[fp,tn]]

K = 2
img = np.array(Image.open('img\experiment\hc\sp_hc2.jpg'))
# 初始化权重矩阵
weight = produce_weight(img, K)
weight_new = np.zeros((img.shape[0], img.shape[1], K))
kexi = np.max(abs(weight - weight_new))
deta = 0.01
t = 0

while kexi>deta:
    center = cal_center(weight, img, K)
    print("t:", t)
    print("center: \n", center)
    g = Gki(img, K, center, weight)
    weight_new = weight
    weight = cal_weight(img, center, g, K)
    kexi = np.max(abs(weight - weight_new))
    t += 1
img = separation(img, weight, center)
plt.figure('FLICM处理图')
plt.title("process_img")
plt.imshow(img,cmap='gray')
plt.show()

# d = DIR(img2, weight)
# print("d",d)
#
# e = e_value(weight,img2)
# print("e:",e)

# 计算合成图的正确率
true_label = np.loadtxt("true_label.txt")
rate = cal_correct(true_label,weight)
print("rate:", rate)

c_matrix = confusion_matrix(true_label,weight)
print("c:",c_matrix)