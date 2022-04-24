import numpy  as np
import os
import random
import cv2
"""
随机挑选CNum张图片，进行按通道计算均值mean和标准差std
先将像素值从0-255归一化到0-1在计算
"""

train_txt_path=os.path.join("/home/dell/桌面/lyl_Person_Reid/yolo_detection/bbox_datasets/","train.txt")
CNum=70

img_h,img_w=128,256
imgs=np.zeros([img_w,img_h,3,1])
means,stdevs=[],[]

with open(train_txt_path,'r') as f:
    lines=f.readlines()
    random.shuffle(lines)
    for i in range(CNum):
        img_path=lines[i].rsplit()[0]
        label=str(lines[i].rsplit()[1])
        img=cv2.imread(img_path)
        img=cv2.resize(img,(img_h,img_w))
        img=img[:,:,:,np.newaxis]
        imgs=np.concatenate((imgs,img),axis=3)

imgs=imgs.astype(np.float32)/255

for i in range(3):
    pixels=imgs[:,:,i,:].ravel() #拉成一行
    means.append(np.mean(pixels))
    stdevs.append(np.std(pixels))

means.reverse() #BGR->RGB
stdevs.reverse()
print(means)
print(stdevs)
print("normMean=".format(means))
print("normStd=".format(stdevs))
