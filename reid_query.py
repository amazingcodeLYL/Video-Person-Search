from os import pardir
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from feature_extract.utils.util import get_distance
from numpy.core.defchararray import title

"""
author: liuyalei
email: liuyalei@mail.ustc.edu.cn
date: 2/12/2021
"""

def imshow(path, title=None):
    im = cv.imread(path)
    im = cv.resize(im,(256,512))
    im = cv.cvtColor(im, cv.COLOR_BGR2RGB)
    plt.imshow(im)
    if title is not None:
        plt.title(title)



# 计算两个图像之间的距离，放到列表里
def rank(probe_ft, gallery_ft, lable, title):

    # import pdb
    # pdb.set_trace()

    distance = []
    oushi = []

    for i in range(len(gallery_ft)):
        temp_dist = get_distance(probe_ft, gallery_ft[i][0])
        distance.append([temp_dist, gallery_ft[i][1], gallery_ft[i][2]])
        oushi.append(temp_dist)
    oushi = np.array(oushi)
    distance_array = np.array(distance)

    dis_T = distance_array.T
    dis = dis_T[0]
    gallery_label = dis_T[1]
    rank_path = dis_T[2]

    dis = dis[oushi.argsort()]
    gallery_label = gallery_label[oushi.argsort()]
    rank_path = rank_path[oushi.argsort()]

    print('距离排序\n', dis[:20])
    print('rank排序\n', rank_path[:10])
    print('gallery lable\n', gallery_label[:10])


    plt.figure(title, figsize=(10,5))
    ax = plt.subplot(1, 11, 1)
    ax.axis('off')
    imshow(title, 'probe')
    for i in range(10):
        ax = plt.subplot(1, 11, i+2)
        ax.axis('off')
        if lable == int(gallery_label[i]):
            ax.set_title('%d' % (i+1), color='green')
        else:
            ax.set_title('%d' % (i+1), color='red')
        imshow(rank_path[i], 'rank'+str(i+1))
    plt.savefig('/ssd/wwz/cv/bishe/lyl_Person_Reid/'+str(lable)+'_ft10.png')

probe_ft=np.load('/ssd/wwz/cv/bishe/lyl_Person_Reid/feature_extract/probeft-2048.npy', allow_pickle=True)
gallery_ft=np.load('/ssd/wwz/cv/bishe/lyl_Person_Reid/feature_extract/galleryft-2048.npy', allow_pickle=True)

for probe in probe_ft:
    rank(probe[0], gallery_ft, probe[1], probe[2])
