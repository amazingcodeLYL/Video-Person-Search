from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
import os
import torch

device = torch.device("cuda:0" if  torch.cuda.is_available() else "cpu")

class MyDataset(Dataset):
    def __init__(self, txt_path, transform=None, target_transform=None) :
        f = open(txt_path,'r')
        imgs = []
        for line in f:
            line = line.rsplit()
            # words = line.split()
            imgs.append((line[0], int(line[1])))
        
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
    
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, index):

        img, label = self.imgs[index]
        img = Image.open(img).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
        return img, label

def validate(net, data_loader, set_name, classes_name):
    """
    对一批数据进行预测，返回混淆矩阵和Accuracy
    :param net:
    :param data_loader:
    :param set_name:
    :param classes_name:
    :return:
    """
    net.eval()
    cls_num = len(classes_name)
    conf_mat = np.zeros([cls_num, cls_num])

    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = net(images)
        outputs.detach_()

        _, predicted = torch.max(outputs.data, 1)

        # 统计混淆矩阵
        for i in range(len(labels)):
            cate_i = labels[i].cpu().numpy()
            pre_i = predicted[i].cpu().numpy()
            conf_mat[cate_i, pre_i] += 1.0

    for i in range(cls_num):
        print('class:{:<10}, total num:{:<6}, correct num:{:<5}, Recall:{:.2%}, Precision:{:.2%}'.format(
            classes_name[i],
            np.sum(np.sum(conf_mat[i, :])),
            conf_mat[i, i],
            conf_mat[i, i] / (1 + np.sum(conf_mat[i, :])),
            conf_mat[i, i] / (1 + np.sum(conf_mat[:, i]))
        ))

    accuracy = np.trace(conf_mat) / np.sum(conf_mat)
    print('{} set Accuracy:{:.2%}'.format(set_name, accuracy))

    return conf_mat, '{:.2}'.format(accuracy)
   

def show_confMat(confusion_mat, classes_name, set_name, out_dir):
    """
    可视化混淆矩阵，保存png格式
    :param confusion_mat: nd-array
    :param classes_name: list,各类别名称
    :param set_name: str, eg: 'valid', 'train'
    :param out_dir: str, png输出的文件夹
    :return:
    """
    # 归一化
    confusion_mat_N = confusion_mat.copy()
    for i in range(len(classes_name)):
        confusion_mat_N[i, :] = confusion_mat[i, :] / confusion_mat[i, :].sum()

    # 获取颜色
    cmap = plt.cm.get_cmap('Blues')  # 更多颜色  http://matplotlib.org/examples/color/colormaps_reference.html
    plt.gcf().subplots_adjust(bottom=0.3)
    plt.imshow(confusion_mat_N, cmap=cmap)
    plt.colorbar()
    plt.rcParams['font.sans-serif'] = ['SimHei'] #显示中文标签 
    plt.rcParams['axes.unicode_minus'] = False
    # 设置文字
    xlocations = np.array(range(len(classes_name)))
    plt.xticks(xlocations, classes_name, rotation=45)
    plt.yticks(xlocations, classes_name)
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.title('Classifier' + set_name)

    

    # 打印数字
    for i in range(confusion_mat_N.shape[0]):
        for j in range(confusion_mat_N.shape[1]):
            plt.text(x=j, y=i, s=round(confusion_mat_N[i, j], 3), va='center', ha='center',color='black', fontsize=7)
            # plt.text(x=j, y=i, s=int(confusion_mat[i, j]), va='center', ha='center', color='red', fontsize=7)
    # 保存
    plt.savefig(os.path.join(out_dir, '混淆矩阵图像' + '.png'), dpi= 600)
    plt.close()

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1) # 归一化，将图片像素压缩在 0 1 之间
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

def get_distance(v1,v2):
    cha = [v1[i] - v2[i] for i in range(len(v1))]
    square = [item * item for item in cha]
    distance = sum(square)
    return distance

def visualize_model(device, model, evalloader, classes_name, num_images=6):
    model.load_state_dict(torch.load('./pretrained/best_model.pth'))
    was_training = model.training
    model.eval()
    fig = plt.figure()
    images_so_far = 0
    with torch.no_grad():
        for i, data in enumerate(evalloader):
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)

            _, preds = torch.max(outputs, 1)

            for j in range(images.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted:{}'.format(classes_name[preds[j]]))
                imshow(images.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return

        model.train(mode=was_training)

