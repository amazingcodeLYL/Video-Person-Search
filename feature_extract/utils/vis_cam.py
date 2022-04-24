# coding: utf-8
import os
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.autograd as autograd
import torchvision.transforms as transforms
import torchvision.models as models
# from senet.se_resnet import FineTuneSEResnet50
import torch.nn as nn
from torch.nn import init

def weights_init_kaiming(m):        #何凯明提出了针对于Relu的初始化方法
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in') # For old pytorch, you may use kaiming_normal.
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)


 # Defines the new fc layer and classification layer
# |--Linear--|--BatchNorm--|--relu--|--Linear--|
class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, droprate, relu=False, bnorm=True, num_bottleneck=512, linear=True, return_f = False):
        super(ClassBlock, self).__init__()
        self.return_f = return_f
        add_block = []
        if linear:
            add_block += [nn.Linear(input_dim, num_bottleneck)]    #对输入数据做线性变换,input_dim, num_bottleneck分别为输入输出神经元的个数
        else:
            num_bottleneck = input_dim
        if bnorm:
            add_block += [nn.BatchNorm1d(num_bottleneck)]       #对小批量(mini-batch)的2d或3d输入进行批标准化，num_bottlenec来自期望输入的特征数
        if relu:
            add_block += [nn.LeakyReLU(0.1)]                    #选用LeakyReLU激活函数
        if droprate>0:
            add_block += [nn.Dropout(p=droprate)]                 #随机将输入张量中部分元素设置为0，dropout操作防止过拟合
        add_block = nn.Sequential(*add_block)                   #一个有序的容器，神经网络模块将按照在传入构造器的顺序依次被添加到计算图中执行
        add_block.apply(weights_init_kaiming)

        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier
    def forward(self, x):
        x = self.add_block(x)
        if self.return_f:
            f = x
            x = self.classifier(x)
            return x,f
        else:
            x = self.classifier(x)
            return x

# Define the ResNet50-based Model
class ft_net(nn.Module):

    def __init__(self, class_num, droprate=0.5, stride=2):
        super(ft_net, self).__init__()
        model_ft = models.resnet50(pretrained=True)
        #print(model_ft)
        # avg pooling to global pooling
        if stride == 1:
            model_ft.layer4[0].downsample[0].stride = (1,1)
            model_ft.layer4[0].conv2.stride = (1,1)
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1,1))         #二维自适应均值池化（一种降维均值池化的方式，池化后的特征图大小为1*1）
        self.model = model_ft
        self.classifier = ClassBlock(2048, class_num, droprate)     #2048为Resnet50最后一层feature map的通道数

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = x.view(x.size(0), x.size(1))
        x = self.classifier(x)
        return x
# 训练过的模型路径
resume_path = r"/ssd/wwz/cv/lyl_Person_Reid/Log/12-06_16-49-18/net_params.pkl"
# 输入图像路径
single_img_path = r'/ssd/wwz/cv/lyl_Person_Reid/yolo_detection/bbox_datasets/1.jpg'
# 绘制的热力图存储路径
save_path = r'/ssd/wwz/cv/lyl_Person_Reid/yolo_detection//temp_layer4.jpg'
 
# 网络层的层名列表, 需要根据实际使用网络进行修改
layers_names = ['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3', 'layer4', 'avgpool']
# 指定层名
out_layer_name = "layer4"
 
features_grad = 0
 
 
# 为了读取模型中间参数变量的梯度而定义的辅助函数
def extract(g):
    global features_grad
    features_grad = g
 
 
def draw_CAM(model, img_path, save_path, transform=None, visual_heatmap=False, out_layer=None):
    """
    绘制 Class Activation Map
    :param model: 加载好权重的Pytorch model
    :param img_path: 测试图片路径
    :param save_path: CAM结果保存路径
    :param transform: 输入图像预处理方法
    :param visual_heatmap: 是否可视化原始heatmap（调用matplotlib）
    :return:
    """
    # 读取图像并预处理
    global layer2
    img = Image.open(img_path).convert('RGB')
    if transform:
        img = transform(img).cuda()
    img = img.unsqueeze(0)  # (1, 3, 448, 448)
 
    # model转为eval模式
    model.eval()
 
    # 获取模型层的字典
    layers_dict = {layers_names[i]: None for i in range(len(layers_names))}
    for i, (name, module) in enumerate(model.features._modules.items()):
        layers_dict[layers_names[i]] = module
 
    # 遍历模型的每一层, 获得指定层的输出特征图
    # features: 指定层输出的特征图, features_flatten: 为继续完成前端传播而设置的变量
    features = img
    start_flatten = False
    features_flatten = None
    for name, layer in layers_dict.items():
        if name != out_layer and start_flatten is False:    # 指定层之前
            features = layer(features)
        elif name == out_layer and start_flatten is False:  # 指定层
            features = layer(features)
            start_flatten = True
        else:   # 指定层之后
            if features_flatten is None:
                features_flatten = layer(features)
            else:
                features_flatten = layer(features_flatten)
 
    features_flatten = torch.flatten(features_flatten, 1)
    output = model.classifier(features_flatten)
 
    # 预测得分最高的那一类对应的输出score
    pred = torch.argmax(output, 1).item()
    pred_class = output[:, pred]
 
    # 求中间变量features的梯度
    # 方法1
    # features.register_hook(extract)
    # pred_class.backward()
    # 方法2
    features_grad = autograd.grad(pred_class, features, allow_unused=True)[0]
 
    grads = features_grad  # 获取梯度
    pooled_grads = torch.nn.functional.adaptive_avg_pool2d(grads, (1, 1))
    # 此处batch size默认为1，所以去掉了第0维（batch size维）
    pooled_grads = pooled_grads[0]
    features = features[0]
    print("pooled_grads:", pooled_grads.shape)
    print("features:", features.shape)
    # features.shape[0]是指定层feature的通道数
    for i in range(features.shape[0]):
        features[i, ...] *= pooled_grads[i, ...]
 
    # 计算heatmap
    heatmap = features.detach().cpu().numpy()
    heatmap = np.mean(heatmap, axis=0)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
 
    # 可视化原始热力图
    if visual_heatmap:
        plt.matshow(heatmap)
        plt.show()
 
    img = cv2.imread(img_path)  # 用cv2加载原始图像
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))  # 将热力图的大小调整为与原始图像相同
    heatmap = np.uint8(255 * heatmap)  # 将热力图转换为RGB格式
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # 将热力图应用于原始图像
    superimposed_img = heatmap * 0.7 + img  # 这里的0.4是热力图强度因子
    cv2.imwrite(save_path, superimposed_img)  # 将图像保存到硬盘
 
 
if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Resize(448),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    classes_name = ['0002', '0007', '0010', '0011', '0012', '0020', '0022', '0023', '0027', '0028', '0030', '0032', '0035', '0037', '0042', '0043', '0046', '0047', '0048', '0052', '0053', '0056', '0057', '0059', '0064', '0065', '0067', '0068', '0069', '0070', '0076', '0077', '0079', '0081', '0082', '0084', '0086', '0088', '0090', '0093', '0095', '0097', '0098', '0099', '0100', '0104', '0105', '0106', '0107', '0108', '0110', '0111', '0114', '0115', '0116', '0117', '0118', '0121', '0122', '0123', '0125', '0127', '0129', '0132', '0134', '0135', '0136', '0139', '0140', '0141', '0142', '0143', '0148', '0149', '0150', '0151', '0158', '0159', '0160', '0162', '0164', '0166', '0167', '0169', '0172', '0173', '0175', '0176', '0177', '0178', '0179', '0180', '0181', '0184', '0185', '0190', '0193', '0195', '0197', '0199', '0201', '0202', '0204', '0206', '0208', '0209', '0211', '0212', '0214', '0216', '0221', '0222', '0223', '0224', '0225', '0232', '0234', '0236', '0237', '0239', '0241', '0242', '0243', '0245', '0248', '0249', '0250', '0251', '0254', '0255', '0259', '0261', '0264', '0266', '0268', '0269', '0272', '0273', '0276', '0277', '0279', '0281', '0282', '0287', '0296', '0297', '0298', '0299', '0301', '0303', '0306', '0307', '0308', '0309', '0313', '0314', '0317', '0318', '0321', '0323', '0324', '0325', '0326', '0327', '0328', '0331', '0332', '0333', '0335', '0338', '0339', '0340', '0341', '0344', '0347', '0348', '0349', '0350', '0352', '0354', '0357', '0358', '0359', '0361', '0367', '0368', '0369', '0370', '0371', '0374', '0375', '0376', '0377', '0379', '0380', '0382', '0383', '0384', '0385', '0386', '0389', '0390', '0392', '0393', '0394', '0397', '0398', '0399', '0402', '0403', '0404', '0407', '0408', '0409', '0410', '0411', '0413', '0414', '0415', '0419', '0420', '0421', '0423', '0424', '0427', '0429', '0430', '0432', '0433', '0434', '0435', '0437', '0441', '0442', '0444', '0445', '0446', '0449', '0450', '0451', '0456', '0457', '0459', '0464', '0466', '0468', '0470', '0472', '0475', '0477', '0480', '0481', '0482', '0484', '0485', '0486', '0491', '0494', '0496', '0499', '0500', '0503', '0508', '0509', '0513', '0515', '0516', '0517', '0518', '0519', '0522', '0524', '0525', '0528', '0529', '0534', '0536', '0537', '0539', '0540', '0545', '0546', '0547', '0549', '0551', '0552', '0554', '0555', '0556', '0557', '0558', '0563', '0564', '0565', '0566', '0570', '0571', '0572', '0573', '0575', '0579', '0581', '0584', '0586', '0588', '0589', '0592', '0593', '0594', '0596', '0597', '0599', '0603', '0604', '0605', '0606', '0611', '0612', '0613', '0614', '0615', '0616', '0619', '0620', '0622', '0623', '0628', '0629', '0630', '0633', '0635', '0636', '0637', '0639', '0640', '0641', '0642', '0645', '0647', '0648', '0649', '0652', '0653', '0655', '0656', '0657', '0658', '0659', '0660', '0661', '0662', '0663', '0665', '0666', '0667', '0669', '0670', '0673', '0674', '0676', '0677', '0681', '0682', '0683', '0685', '0688', '0689', '0696', '0697', '0700', '0701', '0702', '0703', '0704', '0705', '0706', '0707', '0708', '0709', '0711', '0712', '0714', '0718', '0724', '0726', '0729', '0730', '0733', '0734', '0738', '0739', '0741', '0742', '0744', '0748', '0749', '0752', '0754', '0755', '0757', '0759', '0760', '0761', '0762', '0765', '0766', '0767', '0772', '0773', '0774', '0779', '0780', '0781', '0782', '0785', '0787', '0788', '0792', '0793', '0795', '0796', '0802', '0803', '0806', '0809', '0810', '0814', '0816', '0818', '0820', '0821', '0823', '0826', '0828', '0830', '0832', '0833', '0837', '0839', '0840', '0842', '0843', '0844', '0848', '0849', '0850', '0851', '0854', '0855', '0857', '0859', '0862', '0863', '0864', '0868', '0871', '0872', '0875', '0876', '0879', '0882', '0883', '0885', '0886', '0887', '0890', '0891', '0892', '0893', '0894', '0895', '0896', '0898', '0900', '0901', '0902', '0903', '0904', '0905', '0907', '0914', '0915', '0917', '0919', '0926', '0930', '0933', '0936', '0939', '0940', '0941', '0942', '0943', '0945', '0946', '0947', '0948', '0952', '0953', '0954', '0955', '0957', '0958', '0961', '0962', '0963', '0967', '0969', '0970', '0971', '0972', '0973', '0975', '0976', '0979', '0982', '0984', '0986', '0987', '0988', '0990', '0991', '0992', '0994', '0995', '0997', '0998', '0999', '1000', '1001', '1002', '1003', '1004', '1007', '1010', '1011', '1012', '1017', '1018', '1019', '1023', '1025', '1027', '1030', '1031', '1032', '1033', '1038', '1039', '1041', '1045', '1048', '1049', '1051', '1052', '1055', '1056', '1066', '1071', '1072', '1075', '1076', '1078', '1079', '1080', '1081', '1086', '1088', '1091', '1093', '1094', '1096', '1097', '1098', '1099', '1100', '1101', '1106', '1107', '1110', '1111', '1112', '1113', '1114', '1115', '1116', '1117', '1123', '1124', '1126', '1127', '1129', '1132', '1134', '1135', '1138', '1140', '1142', '1152', '1157', '1158', '1159', '1162', '1165', '1167', '1168', '1169', '1173', '1176', '1177', '1178', '1179', '1189', '1193', '1197', '1198', '1200', '1201', '1204', '1206', '1213', '1217', '1218', '1219', '1220', '1227', '1230', '1231', '1232', '1234', '1235', '1237', '1238', '1240', '1242', '1243', '1244', '1250', '1252', '1253', '1254', '1257', '1258', '1260', '1261', '1263', '1266', '1269', '1275', '1278', '1281', '1286', '1289', '1291', '1292', '1294', '1295', '1296', '1297', '1300', '1303', '1304', '1309', '1313', '1315', '1316', '1318', '1320', '1321', '1325', '1326', '1327', '1330', '1331', '1332', '1334', '1335', '1336', '1338', '1339', '1341', '1343', '1344', '1346', '1350', '1353', '1358', '1363', '1364', '1365', '1368', '1372', '1373', '1379', '1380', '1381', '1385', '1386', '1389', '1391', '1392', '1393', '1400', '1402', '1404', '1405', '1406', '1407', '1408', '1409', '1411', '1415', '1420', '1421', '1422', '1426', '1427', '1428', '1430', '1432', '1433', '1434', '1437', '1442', '1443', '1445', '1447', '1449', '1451', '1453', '1454', '1455', '1458', '1463', '1464', '1466', '1467', '1469', '1470', '1471', '1473', '1474', '1475', '1479', '1480', '1487', '1489', '1492', '1495', '1496', '1500']

    # 构建模型并加载预训练参数
    # seresnet50 = FineTuneSEResnet50(num_class=113).cuda()
    
    pretrained_path = os.path.join("/ssd/wwz/cv/lyl_Person_Reid/", "Log", "12-06_16-49-18", "net_params.pkl")
    # pretrained_dict = torch.load(pretrained_path)
    # net = ft_net(class_num=len(classes_name), droprate=1, stride=2)
    # net.load_state_dict(pretrained_dict)
    seresnet50 = models.resnet50(pretrained=True).cuda()
    # checkpoint = torch.load(resume_path)
    # seresnet50.load_state_dict(checkpoint['state_dict'])
    draw_CAM(seresnet50, single_img_path, save_path, transform=transform, visual_heatmap=True, out_layer=out_layer_name)