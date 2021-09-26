import math
import random

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


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
def init_center(data):
    x = random.randint(0,data.shape[0])
    y = random.randint(0,data.shape[1])
    tmp = data[x][y]
    return tmp
def produce_center(data, K):
    data_size = 1
    a = np.ones((K,data_size))
    for k in range(K):
        center = init_center(data)
        a[k] = center
    return a
def Wj(data):
    row = data.shape[0]
    column = data.shape[1]
    dx = [-1, 0, 1, -1, 1, -1, 0, 1]
    dy = [-1, -1, -1, 0, 0, 1, 1, 1]
    d = [1.414, 1, 1414, 1, 1, 1.414, 1, 1.414]
    sum_neighbor = np.zeros((row, column))
    c = sum_neighbor
    mean_neighbor = np.zeros((row, column))
    var_neighbor = np.zeros((row, column))
    for i in range(1, row-1):
        for j in range(1, column-1):
            t = []
            for r in range(8):
                index_x = i + dx[r]
                index_y = j + dy[r]
                sum_neighbor[i][j] += data[index_x][index_y]
                t.append(data[index_x][index_y])
            var_neighbor[i][j] = np.var(t)
            mean_neighbor[i][j] = sum_neighbor[i][j]/8
            c[i][j] = var_neighbor[i][j]/(pow(mean_neighbor[i][j], 2) + 0.01)
    # print("c shape:", c.shape)
    c_means = np.ones((c.shape))
    for i in range(1, row-1):
        for j in range(1, column-1):
            tmp = 0
            for r in range(8):
                index_x = i + dx[r]
                index_y = j + dy[r]
                tmp += c[index_x][index_y]
            c_means[i][j] = tmp/8
    kexi = np.zeros((row, column, 8))
    for i in range(1, row-1):
        for j in range(1, column-1):
            for r in range(8):
                index_x = i + dx[r]
                index_y = j + dy[r]
                kexi[i][j][r] += np.exp(-(c[index_x][index_y]-c_means[i][j]))
    ita = np.zeros((row, column, 8))
    wgc = np.zeros((row, column, 8))
    wsc = np.zeros((row, column, 8))
    w = np.ones((row, column, 8))
    for i in range(1, row-1):
        for j in range(1, column-1):
            t = 0
            for r in range(8):
                t += kexi[i][j][r]
                wsc[i][j][r] = 1/(d[r] + 1)
            for k in range(8):
                index_x = i + dx[k]
                index_y = j + dy[k]
                ita[i][j][k] = kexi[i][j][k]/t
                if c[index_x][index_y]<c_means[i][j]:
                    wgc[i][j][k] = 2 + ita[i][j][k]
                else:
                    wgc[i][j][k] = 2 - ita[i][j][k]
    for i in range(1, row-1):
        for j in range(1, column-1):
            for r in range(8):
                w[i][j][r] = wsc[i][j][r]*wgc[i][j][r]
    return w
def cal_weight(data, center, weight, w, dist, m = 2):
    row = data.shape[0]
    column = data.shape[1]
    K = center.shape[0]
    miu_factors = np.ones((row, column, K, K))
    dx = [-1, 0, 1, -1, 1, -1, 0, 1]
    dy = [-1, -1, -1, 0, 0, 1, 1, 1]
    for i in range(1, row-1):
        for j in range(1, column-1):
            for k in range(K):
                for l in range(K):
                    t_factor = 0
                    for r in range(8):
                        index_x = i + dx[r]
                        index_y = j + dy[r]
                        t_factor += w[i][j][r] * pow((1-weight[index_x][index_y][k]), m) * (1- dist[index_x][index_y][l] )
                    miu_factors[i][j][k][l] = (1- dist[i][j][l] ) + t_factor
    miu = np.zeros((row, column, K))
    for i in range(row):
        for j in range(column):
            for k in range(K):
                for l in range(K):
                    miu[i][j][k] += pow(miu_factors[i][j][k][k]/miu_factors[i][j][k][l], 1/(m-1))
                miu[i][j][k] = 1/miu[i][j][k]

    return miu
def cal_center(data, center, weight,  sita, m = 2):
    row = data.shape[0]
    column = data.shape[1]
    K = center.shape[0]
    miu_mul_dist = np.ones((row, column, K))
    for k in range(K):
        c_z = c_m = 0
        for i in range(1, row-1):
            for j in range(1, column-1):
                miu_mul_dist[i][j][k] = pow(weight[i][j][k], m) * (grbf(data[i][j], center[k], sita))
                c_z += miu_mul_dist[i][j][k] * data[i][j]
                c_m += miu_mul_dist[i][j][k]
        center[k] = c_z/c_m
    return center
def cal_dist(data, center, sita):
    row = data.shape[0]
    column = data.shape[1]
    distance = np.zeros((row, column, K))
    for i in range(row):
        for j in range(column):
            for k in range(K):
                distance[i][j][k] = grbf(img[i][j], center[k], sita)
    return distance
def cal_sita(data):
    row = data.shape[0]
    column = data.shape[1]
    t = 0
    for i in range(row):
        for j in range(column):
            t += data[i][j]
    data_mean = t/(row * column)
    dist = np.zeros((data.shape))
    for i in range(row):
        for j in range(column):
            dist[i][j] = np.abs(data[i][j] - data_mean)
    return np.var(dist)
# 高斯核
def grbf(x, y, sita = 1569, a = 1, b = 1):
    return np.exp(-pow(pow(abs(x - y), a), b)/sita)
def separation(data, weight,center):
    index = 0
    for i in range(weight.shape[0]):
        for j in range(weight.shape[1]):
            tmp = 0
            for k in range(weight.shape[2]):
                if weight[i][j][k]>=tmp:
                    tmp = weight[i][j][k]
                    index = k
            data[i][j] = center[index]
    return data
def get_label(weight):
    index = 0
    row,column,K = weight.shape
    label = np.zeros((row, column))
    for i in range(row):
        for j in range(column):
            tmp = 0
            for k in range(K):
                if weight[i][j][k]>=tmp:
                    tmp = weight[i][j][k]
                    index = k
            # print("index: ",index)
            label[i][j] = int(index)
    return label
def SA(weight, label):#有监督评价指标
    row,column,K = weight.shape
    cal_label = np.zeros(label.shape)
    cnt = 0
    for i in range(row):
        for j in range(column):
            tmp = index = 0
            for k in range(K):
                if weight[i][j][k]>=tmp:
                    index = k
                cal_label[i][j] = index
            if cal_label[i][j] == label[i][j]:
                cnt += 1
    return cnt/(row*column)
# 评价指标
# def regional_entropy(data,weight):
#     index = 0
#     label = np.zeros(data.shape)
#     for i in range(weight.shape[0]):
#         for j in range(weight.shape[1]):
#             tmp = 0
#             for k in range(weight.shape[2]):
#                 if weight[i][j][k]>=tmp:
#                     tmp = weight[i][j][k]
#                     index = k
#             label[i][j] = index
#     LS = np.zeros((weight.shape[2], 255))
#     row, column = data.shape
#     K = weight.shape[2]
#     for i in range(row):
#         for j in range(column):
#             LS[label[i][j]][data[i][j]] += 1
#     h = np.zeros((1, K))
#     s = np.sum(LS, axis=1)
#     for i in range(LS.shape[0]):
#         for j in range(LS.shape[1]):
#             h[0][i] += -LS[i][j]/s[i]*np.log(LS[i][j]/s[i])
#     return h
# 基于熵的评价
def e_value(weight,data):
    index = 0
    hr = hl = 0
    K = weight.shape[2]
    # cnt = np.zeros((K, 1))
    cnt = [0] * K #记录每类的像素点数目
    label = np.zeros(data.shape)#每个像素点的类别（根据隶属度确定）
    # h = np.zeros((1, K))
    h = [0] * K# 区域熵
    LS = np.zeros((weight.shape[2], 256))#每一类里 像素等于某个灰度值的像素点数目
    row, column = data.shape
    for i in range(weight.shape[0]):
        for j in range(weight.shape[1]):
            tmp = 0
            for k in range(K):
                if weight[i][j][k] >= tmp:
                    tmp = weight[i][j][k]
                    index = k
            cnt[index] += 1
            label[i][j] = index
    for i in range(row):
        for j in range(column):
            x = int(label[i][j])
            y = int(data[i][j])
            LS[x][y] += 1
    s = np.sum(LS, axis=1)
    # print("s: ",s)
    for i in range(LS.shape[0]):
        for j in range(LS.shape[1]):
            if LS[i][j] != 0:
                h[i] += -LS[i][j]/(s[i])*np.log(LS[i][j]/(s[i]))
    for k in range(K):
        hr += (cnt[k]/(row*column))*h[k]
        hl += -(cnt[k]/(row*column))*np.log((cnt[k]/(row*column)))
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

img = np.array(Image.open('timg2.jpg'))
K = 2
sita = cal_sita(img)
w = Wj(img)
weight = produce_weight(img, K)
center = produce_center(img, K)
# center = np.array([[208.],[128.]]) #如多次实验不能正常运行可使用这行代码

weight_new = np.ones((img.shape[0], img.shape[1], K))
kexi = np.max(abs(weight - weight_new))
while math.isnan(kexi):
    weight = produce_weight(img, K)
    weight_new = np.ones((img.shape[0], img.shape[1], K))
    kexi = np.max(abs(weight - weight_new))
deta = 0.01
t = 0

while kexi>deta and t<=20:
    print("t:", t)
    print("center: \n", center)
    weight_new = weight
    d = cal_dist(img, center, sita)
    weight = cal_weight(img, center, weight, w, d)
    center = cal_center(img, center, weight, sita)
    kexi = np.max(abs(weight - weight_new))
    t += 1

img = separation(img, weight, center)

plt.figure('KWFLICM处理图')
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