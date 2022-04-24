from tensorboardX.summary import image
import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
import numpy as np
import os
import argparse
# from torch.autograd import Variable
from model import ft_net
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
sys.path.append("..")
from utils.util import MyDataset, validate, show_confMat,imshow
from tensorboardX import SummaryWriter
from datetime import datetime

device = torch.device("cuda:0" if  torch.cuda.is_available() else "cpu")

classes_name = ['0002', '0007', '0010', '0011', '0012', '0020', '0022', '0023', '0027', '0028', '0030', '0032', '0035', '0037', '0042', '0043', '0046', '0047', '0048', '0052', '0053', '0056', '0057', '0059', '0064', '0065', '0067', '0068', '0069', '0070', '0076', '0077', '0079', '0081', '0082', '0084', '0086', '0088', '0090', '0093', '0095', '0097', '0098', '0099', '0100', '0104', '0105', '0106', '0107', '0108', '0110', '0111', '0114', '0115', '0116', '0117', '0118', '0121', '0122', '0123', '0125', '0127', '0129', '0132', '0134', '0135', '0136', '0139', '0140', '0141', '0142', '0143', '0148', '0149', '0150', '0151', '0158', '0159', '0160', '0162', '0164', '0166', '0167', '0169', '0172', '0173', '0175', '0176', '0177', '0178', '0179', '0180', '0181', '0184', '0185', '0190', '0193', '0195', '0197', '0199', '0201', '0202', '0204', '0206', '0208', '0209', '0211', '0212', '0214', '0216', '0221', '0222', '0223', '0224', '0225', '0232', '0234', '0236', '0237', '0239', '0241', '0242', '0243', '0245', '0248', '0249', '0250', '0251', '0254', '0255', '0259', '0261', '0264', '0266', '0268', '0269', '0272', '0273', '0276', '0277', '0279', '0281', '0282', '0287', '0296', '0297', '0298', '0299', '0301', '0303', '0306', '0307', '0308', '0309', '0313', '0314', '0317', '0318', '0321', '0323', '0324', '0325', '0326', '0327', '0328', '0331', '0332', '0333', '0335', '0338', '0339', '0340', '0341', '0344', '0347', '0348', '0349', '0350', '0352', '0354', '0357', '0358', '0359', '0361', '0367', '0368', '0369', '0370', '0371', '0374', '0375', '0376', '0377', '0379', '0380', '0382', '0383', '0384', '0385', '0386', '0389', '0390', '0392', '0393', '0394', '0397', '0398', '0399', '0402', '0403', '0404', '0407', '0408', '0409', '0410', '0411', '0413', '0414', '0415', '0419', '0420', '0421', '0423', '0424', '0427', '0429', '0430', '0432', '0433', '0434', '0435', '0437', '0441', '0442', '0444', '0445', '0446', '0449', '0450', '0451', '0456', '0457', '0459', '0464', '0466', '0468', '0470', '0472', '0475', '0477', '0480', '0481', '0482', '0484', '0485', '0486', '0491', '0494', '0496', '0499', '0500', '0503', '0508', '0509', '0513', '0515', '0516', '0517', '0518', '0519', '0522', '0524', '0525', '0528', '0529', '0534', '0536', '0537', '0539', '0540', '0545', '0546', '0547', '0549', '0551', '0552', '0554', '0555', '0556', '0557', '0558', '0563', '0564', '0565', '0566', '0570', '0571', '0572', '0573', '0575', '0579', '0581', '0584', '0586', '0588', '0589', '0592', '0593', '0594', '0596', '0597', '0599', '0603', '0604', '0605', '0606', '0611', '0612', '0613', '0614', '0615', '0616', '0619', '0620', '0622', '0623', '0628', '0629', '0630', '0633', '0635', '0636', '0637', '0639', '0640', '0641', '0642', '0645', '0647', '0648', '0649', '0652', '0653', '0655', '0656', '0657', '0658', '0659', '0660', '0661', '0662', '0663', '0665', '0666', '0667', '0669', '0670', '0673', '0674', '0676', '0677', '0681', '0682', '0683', '0685', '0688', '0689', '0696', '0697', '0700', '0701', '0702', '0703', '0704', '0705', '0706', '0707', '0708', '0709', '0711', '0712', '0714', '0718', '0724', '0726', '0729', '0730', '0733', '0734', '0738', '0739', '0741', '0742', '0744', '0748', '0749', '0752', '0754', '0755', '0757', '0759', '0760', '0761', '0762', '0765', '0766', '0767', '0772', '0773', '0774', '0779', '0780', '0781', '0782', '0785', '0787', '0788', '0792', '0793', '0795', '0796', '0802', '0803', '0806', '0809', '0810', '0814', '0816', '0818', '0820', '0821', '0823', '0826', '0828', '0830', '0832', '0833', '0837', '0839', '0840', '0842', '0843', '0844', '0848', '0849', '0850', '0851', '0854', '0855', '0857', '0859', '0862', '0863', '0864', '0868', '0871', '0872', '0875', '0876', '0879', '0882', '0883', '0885', '0886', '0887', '0890', '0891', '0892', '0893', '0894', '0895', '0896', '0898', '0900', '0901', '0902', '0903', '0904', '0905', '0907', '0914', '0915', '0917', '0919', '0926', '0930', '0933', '0936', '0939', '0940', '0941', '0942', '0943', '0945', '0946', '0947', '0948', '0952', '0953', '0954', '0955', '0957', '0958', '0961', '0962', '0963', '0967', '0969', '0970', '0971', '0972', '0973', '0975', '0976', '0979', '0982', '0984', '0986', '0987', '0988', '0990', '0991', '0992', '0994', '0995', '0997', '0998', '0999', '1000', '1001', '1002', '1003', '1004', '1007', '1010', '1011', '1012', '1017', '1018', '1019', '1023', '1025', '1027', '1030', '1031', '1032', '1033', '1038', '1039', '1041', '1045', '1048', '1049', '1051', '1052', '1055', '1056', '1066', '1071', '1072', '1075', '1076', '1078', '1079', '1080', '1081', '1086', '1088', '1091', '1093', '1094', '1096', '1097', '1098', '1099', '1100', '1101', '1106', '1107', '1110', '1111', '1112', '1113', '1114', '1115', '1116', '1117', '1123', '1124', '1126', '1127', '1129', '1132', '1134', '1135', '1138', '1140', '1142', '1152', '1157', '1158', '1159', '1162', '1165', '1167', '1168', '1169', '1173', '1176', '1177', '1178', '1179', '1189', '1193', '1197', '1198', '1200', '1201', '1204', '1206', '1213', '1217', '1218', '1219', '1220', '1227', '1230', '1231', '1232', '1234', '1235', '1237', '1238', '1240', '1242', '1243', '1244', '1250', '1252', '1253', '1254', '1257', '1258', '1260', '1261', '1263', '1266', '1269', '1275', '1278', '1281', '1286', '1289', '1291', '1292', '1294', '1295', '1296', '1297', '1300', '1303', '1304', '1309', '1313', '1315', '1316', '1318', '1320', '1321', '1325', '1326', '1327', '1330', '1331', '1332', '1334', '1335', '1336', '1338', '1339', '1341', '1343', '1344', '1346', '1350', '1353', '1358', '1363', '1364', '1365', '1368', '1372', '1373', '1379', '1380', '1381', '1385', '1386', '1389', '1391', '1392', '1393', '1400', '1402', '1404', '1405', '1406', '1407', '1408', '1409', '1411', '1415', '1420', '1421', '1422', '1426', '1427', '1428', '1430', '1432', '1433', '1434', '1437', '1442', '1443', '1445', '1447', '1449', '1451', '1453', '1454', '1455', '1458', '1463', '1464', '1466', '1467', '1469', '1470', '1471', '1473', '1474', '1475', '1479', '1480', '1487', '1489', '1492', '1495', '1496', '1500']
lr_init = 0.0005
max_epoch = 35

now_time = datetime.now()
time_str = datetime.strftime(now_time, '%m-%d_%H-%M-%S')
# log
result_dir = os.path.join("/ssd/wwz/cv/bishe/lyl_Person_Reid/", "Log")
log_dir = os.path.join(result_dir, time_str)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

writer = SummaryWriter(log_dir=log_dir)
# ------------------------------------ step 1/5 : 加载数据------------------------------------

# 数据预处理设置
# normMean = [0.3858418, 0.3703364, 0.37060848]
# normStd = [0.2820569, 0.28218642, 0.28078946]

parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--data_dir',default='/ssd/wwz/cv/bishe/lyl_Person_Reid/feature_extract/Market-1501-v15.09.15/pytorch',type=str, help='training dir path')
parser.add_argument('--train_all', action='store_true', help='use all training data' )
parser.add_argument('--batchsize', default=32, type=int, help='batchsize')
opt = parser.parse_args()
data_dir = opt.data_dir
train_all = ''
if opt.train_all:
     train_all = '_all'


transform_train_list = [
        #transforms.RandomResizedCrop(size=128, scale=(0.75,1.0), ratio=(0.75,1.3333), interpolation=3), #Image.BICUBIC)
        transforms.Resize((256,128), interpolation=3),
        transforms.Pad(10),
        transforms.RandomCrop((256,128)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]

transform_val_list = [
        transforms.Resize(size=(256,128),interpolation=3), #Image.BICUBIC
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
data_transforms = {
    'train': transforms.Compose( transform_train_list ),
    'val': transforms.Compose(transform_val_list),
}
image_datasets = {}
image_datasets['train'] = datasets.ImageFolder(os.path.join(data_dir, 'train' + train_all),
                                          data_transforms['train'])
image_datasets['val'] = datasets.ImageFolder(os.path.join(data_dir, 'val'),
                                          data_transforms['val'])

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                             shuffle=True, num_workers=16)
              for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

# 显示一个batch的图像
# images, labels = next(iter(train_loader))
# print(labels)
# out=torchvision.utils.make_grid(images)
# imshow(out, title=[classes_name[x] for x in labels])

# ------------------------------------ step 2/5 : 定义网络------------------------------------
droprate = 0.5
stride = 2
net = ft_net(len(classes_name), droprate, stride)     # 创建一个网络
net.to(device)
# net.initialize_weights()    # 初始化权值

# ------------------------------------ step 3/5 : 定义损失函数和优化器 ------------------------------------

criterion = nn.CrossEntropyLoss()                                                   
optimizer = optim.SGD(net.parameters(), lr=lr_init, momentum=0.9, dampening=0.1)    
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)     

# ------------------------------------ step 4/5 : 训练 --------------------------------------------------
with open(log_dir+"/log.txt", "w") as f:
    for epoch in range(max_epoch):
        loss_sigma = 0.0    # 记录一个epoch的loss之和
        correct = 0.0
        total = 0.0
        scheduler.step()  # 更新学习率
        for i, data in enumerate(dataloaders['train']):
            # 获取图片和标签
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device) 

            # forward, backward, update weights
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # 统计预测信息
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).squeeze().sum().cpu().numpy()
            loss_sigma += loss.item()

            # 每10个iteration 打印一次训练信息，loss为10个iteration的平均
            if i % 10 == 9:
                loss_avg = loss_sigma / 10
                loss_sigma = 0.0
                print("Training: Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}".format(
                    epoch + 1, max_epoch, i + 1, len(dataloaders['train']), loss_avg, correct / total))
                f.write("Training: Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}\n".format(
                    epoch + 1, max_epoch, i + 1, len(dataloaders['train']), loss_avg, correct / total))

                # 记录训练loss
                writer.add_scalars('Loss_group', {'train_loss': loss_avg}, epoch)
                # 记录learning rate
                writer.add_scalar('learning rate', scheduler.get_lr()[0], epoch)
                # 记录Accuracy
                writer.add_scalars('Accuracy_group', {'train_acc': correct / total}, epoch)

        # 每个epoch，记录梯度，权值
        # for name, layer in net.named_parameters():
        #     writer.add_histogram(name + '_grad', layer.grad.cpu().data.numpy(), epoch)
        #     writer.add_histogram(name + '_data', layer.cpu().data.numpy(), epoch)

        # ------------------------------------ 观察模型在验证集上的表现 ------------------------------------
        if epoch % 2 == 0:
            loss_sigma_val = 0.0
            correct_val = 0.0
            total_val = 0.0
            cls_num = len(classes_name)
            conf_mat = np.zeros([cls_num, cls_num])  # 混淆矩阵
            net.eval()
            with torch.no_grad():
                # for i, data in enumerate(valid_loader):
                for i, data in enumerate(dataloaders['val']):    
                    # 获取图片和标签
                    images, labels_v = data
                    images, labels_v = images.to(device), labels_v.to(device)

                    # forward
                    outputs_v = net(images)
                    # outputs_v.detach_()

                    # 计算loss
                    loss_val = criterion(outputs_v, labels_v)
                    loss_sigma_val += loss_val.item()

                    # 统计
                    _, predicted_v = torch.max(outputs_v.data, 1)
                    # labels = labels.data    # Variable --> tensor
                    total_val += labels_v.size(0)
                    correct_val += (predicted_v == labels_v).squeeze().sum().cpu().numpy()

                    # 统计混淆矩阵
                    for j in range(len(labels_v)):
                        cate_i = labels_v[j].cpu().numpy()
                        pre_i = predicted_v[j].cpu().numpy()
                        conf_mat[cate_i, pre_i] += 1.0

            # print('{} set Accuracy:{:.2%}'.format('Valid', conf_mat.trace() / conf_mat.sum()))
            print('{} set Accuracy:{:.2%}'.format('Valid', correct_val / total_val))
            f.write('{} set Accuracy:{:.2%}\n'.format('Valid', conf_mat.trace() / conf_mat.sum()))
            # 记录Loss, accuracy
            writer.add_scalars('Loss_group', {'valid_loss': loss_sigma_val / len(dataloaders['val'])}, epoch)
            writer.add_scalars('Accuracy_group', {'valid_acc': conf_mat.trace() / conf_mat.sum()}, epoch)
print('Finished Training')

# ------------------------------------ step5: 保存模型 并且绘制混淆矩阵图 ------------------------------------
net_save_path = os.path.join(log_dir, 'net_params.pkl')
torch.save(net.state_dict(), net_save_path)

conf_mat_train, train_acc = validate(net, dataloaders['train'], 'train', classes_name)
conf_mat_valid, valid_acc = validate(net, dataloaders['val'], 'valid', classes_name)

# show_confMat(conf_mat_train, classes_name, 'train', log_dir)
# show_confMat(conf_mat_valid, classes_name, 'valid', log_dir)

f.close()