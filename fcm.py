import random

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


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

def init_weight(K,data): # K: the number of cluster center
    weight = np.zeros((data.shape[0],data.shape[1],K), dtype = float, order = 'C')
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            for k in range(K):
                weight[i][j][k] = random.random()
    return weight

def cal_weight(K,data,center,weight):
    p = 2
    distance = np.ones((data.shape[0], data.shape[1],K))
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            for k in range(K):
                distance[i][j][k] = pow( 1/(pow(np.linalg.norm(data[i][j] - center[k]), 2)+0.01), 1/(p-1))

    dist_sum = np.ones((data.shape[0],data.shape[1]))
    dist_sum = np.sum(distance,axis=2)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            for k in range(K):
                weight[i][j][k] = distance[i][j][k]/dist_sum[i][j]
    return weight

def cal_center(data,weight,center):
    data_mul_weight = np.zeros((center.shape[0],1))
    weight_sum = np.zeros((1,center.shape[0]))
    for k in range(weight.shape[2]):
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                data_mul_weight[k] += weight[i][j][k]*data[i][j]

    weight_sum = np.sum(np.sum(weight, axis=1),axis=0)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            for k in range(center.shape[0]):
                center[k] = data_mul_weight[k]/weight_sum[k]

    return center
# 对结果进行划分，对于同一类的像素点赋予相同的值
def separation(data,weight,center):
    for i in range(weight.shape[0]):
        for j in range(weight.shape[1]):
            tmp = 0
            label = 0
            for k in range(weight.shape[2]):
                if weight[i][j][k]>=tmp:
                    tmp = weight[i][j][k]
                    label = k
            data[i][j] = center[label][0]

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
    return abs(mean_data[0]-mean_data[1])/(mx - mn) #

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
    print("cnt:",cnt)
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


# 读取图片
# img = np.array(Image.open('img\experiment\hc\sp_hc2.jpg'))
img = np.array(Image.open('blb2.jpg'))
# img2 = np.array(Image.open('img\experiment\MR1\gs_MR11.jpg'))
K = 2 # 类别数
# img = np.array(Image.open('img\experiment\coins\gray_coins.jpg'))
# img = np.array(Image.open('img\experiment\coins\gs_coins1.jpg'))
# img = np.array(Image.open('img\experiment\coins\gs_coins2.jpg'))
# img = np.array(Image.open('img\experiment\coins\sp_coins1.jpg'))
# img = np.array(Image.open('img\experiment\coins\sp_coins2.jpg'))

# img = np.array(Image.open('img\experiment\hc\gray_hc.jpg'))
# img = np.array(Image.open('img\experiment\hc\gs_hc1.jpg'))
# img = np.array(Image.open('img\experiment\hc\gs_hc2.jpg'))
# img = np.array(Image.open('img\experiment\hc\sp_hc1.jpg'))
# img = np.array(Image.open('img\experiment\hc\sp_hc2.jpg'))

# img = np.array(Image.open('img\experiment\MR1\gray_MR1.jpg'))
# img = np.array(Image.open('img\experiment\MR1\gs_MR11.jpg'))
# img = np.array(Image.open('img\experiment\MR1\gs_MR12.jpg'))
# img = np.array(Image.open('img\experiment\MR1\sp_MR11.jpg'))
# img = np.array(Image.open('img\experiment\MR1\sp_MR12.jpg'))

# img = np.array(Image.open('img\experiment\sys\gray_sys.jpg'))
# img = np.array(Image.open('img\experiment\sys\gs_sys1.jpg'))
# img = np.array(Image.open('img\experiment\sys\gs_sys2.jpg'))
# img = np.array(Image.open('img\experiment\sys\sp_sys1.jpg'))
# img = np.array(Image.open('img\experiment\sys\sp_sys2.jpg'))

# img = np.array(Image.open('img\experiment\MR2\gray_MR2.jpg'))
# img = np.array(Image.open('img\experiment\MR2\gs_MR21.jpg'))
# img = np.array(Image.open('img\experiment\MR2\gs_MR22.jpg'))
# img = np.array(Image.open('img\experiment\MR2\sp_MR21.jpg'))

# 初始化中心和权重矩阵
center = produce_center(K, 1, img)
print("center: \n",center)
weight = init_weight(K,img)
T = 10
for t in range(T):
#   更新中心和权重矩阵
    weight = cal_weight(K,img,center,weight)
    center = cal_center(img,weight,center)
    print("t:", t)
    print("center: \n", center)
separation(img,weight,center)
plt.figure('FCM处理图')
plt.title("process_img")
plt.imshow(img,cmap='gray')
plt.show()

# d = DIR(img2, weight)
# print("d",d)
#
# e = e_value(weight,img2)
# print("e:",e)

# 合成图评价标准
# true_label = np.loadtxt("true_label.txt")# 合成图真实类别标签
# rate = cal_correct(true_label,weight) #合成图 有监督正确率
# print("rate:", rate)
#
# c_matrix = confusion_matrix(true_label,weight)
# print("c:",c_matrix)