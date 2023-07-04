import torch.nn as nn
import torch.nn.functional as F
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# reference: https://github.com/kuangliu/pytorch-cifar

import numpy as np
import cvxpy as cp
import csv
from cvxpy import log_sum_exp, sum, Minimize, Problem
from scipy import spatial
import torch

def cross_entropy_loss(logits, label):
    return -logits[label] + log_sum_exp(logits)

def get_total_loss(X, c, t1, t2, R):
    logits = X[0, c:]
    label = 0
    total_loss = cross_entropy_loss(logits, label)
    for i in range(t1)[1:]:
        logits = X[i, c:]
        label = i
        total_loss += cross_entropy_loss(logits, label)
    if R != float('inf'):
        for j in range(t2):
            logits = X[t1 + j, c:]
            label = t1 + j
            total_loss += (cross_entropy_loss(logits, label) / R)
    return total_loss

def get_constraints(X, c, Af, Aw):
    constraints = []
    constraints += [sum([X[i, i] for i in range(c)]) <= c * Af]
    constraints += [sum([X[c + j, c + j] for j in range(c)]) <= c * Aw]
    constraints += [X >> 0]
    # X >= 0
    return constraints

def neural_collapse_optimization(class_num, big_class_num, ratio, feature_constant, weight_constant):
    c = class_num
    t1 = big_class_num
    t2 = c - t1
    R = ratio
    Af = feature_constant
    Aw = weight_constant
    global X
    X = cp.Variable((2 * c, 2 * c), symmetric=True)
    total_loss = get_total_loss(X, c, t1, t2, R)
    constraints = get_constraints(X, c, Af, Aw)
    obj = Minimize(total_loss)
    prob = Problem(obj, constraints)
    try:
        prob.solve()
    except Exception as e:
        print(e)

    X_round = []
    for i in range(len(X.value)):
        X_round.append([round(X.value[i][j], 3) for j in range(len(X.value[0]))])
    # print(X_round)
    with open('tmp_matrix.csv', 'w') as f:
        writer = csv.writer(f)
        for i in range(len(X_round)):
            writer.writerow(X_round[i])
    between_class_cos_small = []
    for i in range(2 * c)[c + t1: ]:
        for j in range(2 * c)[c + t1:]:
            if i != j:
                cos_value = X.value[i, j] / np.sqrt(X.value[i, i] * X.value[j, j])
                between_class_cos_small.append(cos_value)
    return np.mean(between_class_cos_small)

def run_optimization_experiments():
    class_num = 10
    big_class_num = 5
    feature_constant = 100
    weight_constant = 100
    #ratio_list = [np.power(10, i * 0.1) for i in range(41)]
    ratio_list = [50]
    cos_values = []

    for x in range(len(ratio_list)):
        ratio = ratio_list[x]
        cos_values.append(neural_collapse_optimization(class_num, big_class_num, ratio,feature_constant, weight_constant))

run_optimization_experiments()

X_tensor = torch.from_numpy(X.value).float()
eigvals, eigvecs = torch.linalg.eigh(X_tensor)
sqrt_eigvals = torch.sqrt(torch.abs(eigvals))
sqrt_matrix = eigvecs @ torch.diag(sqrt_eigvals) @ eigvecs.T
sqrt_matrix = sqrt_matrix.to(device)

class nonETF_Classifier(nn.Module):
    def __init__(self, feat_in, num_classes):
        super(nonETF_Classifier, self).__init__()

        P = self.generate_random_orthogonal_matrix(feat_in, 2*num_classes)
        P = P.to(device)
        I = torch.eye(num_classes)
        one = torch.ones(num_classes, num_classes)

        PM = torch.matmul(P, sqrt_matrix)
        M = PM[:, num_classes:]

        self.M_ = M.cuda()
        self.ori_M = nn.Parameter(self.M_, requires_grad=False)  # 设置为不可训练

    def generate_random_orthogonal_matrix(self, feat_in, num_classes):
        a = np.random.random(size=(feat_in, num_classes))
        P, _ = np.linalg.qr(a)
        P = torch.tensor(P).float()
        assert torch.allclose(torch.matmul(P.T, P), torch.eye(num_classes), atol=1e-07), torch.max(torch.abs(torch.matmul(P.T, P) - torch.eye(num_classes)))
        return P

    def forward(self, x):
        x = torch.matmul(x, self.ori_M)
        x = x.to(device)
        return x

#class MLP(nn.Module):
#    def __init__(self, input_size, hidden_size, num_classes):
#        super(MLP, self).__init__()
#        self.fc_layers = nn.Sequential(
#            nn.Linear(input_size, hidden_size, bias=False),
#            nn.ReLU(),
#            nn.Linear(hidden_size, hidden_size, bias=False),
#            nn.ReLU(),
#        )
#        self.classifier = ETF_Classifier(hidden_size, num_classes)
#
#    def forward(self, x):
#        x = x.view(x.size(0), -1)
#        features = self.fc_layers(x)
#        logits = self.classifier(features)
#        return logits


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, color_channel=3, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(color_channel, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.classifier = nn.Linear(512*block.expansion, num_classes)
#        self.classifier = nn.Parameter(torch.randn(512*block.expansion, num_classes))
#        self.classifier.requires_grad = False
    

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        #out = torch.matmul(out, self.classifier)
        return out

    def get_features(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        return out

class nonetfResNet(nn.Module):
    def __init__(self, block, num_blocks, color_channel=3, num_classes=10):
        super(nonetfResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(color_channel, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        #self.classifier = nn.Linear(512*block.expansion, num_classes)
#        self.classifier = nn.Parameter(torch.randn(512*block.expansion, num_classes))
#        self.classifier.requires_grad = False
        self.classifier = nonETF_Classifier(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        #out = self.classifier(out)
#        out = torch.matmul(out, self.classifier)
        out = self.classifier(out)
        return out

    def get_features(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        return out


def ResNet18(color_channel=3, num_classes=10):
    return ResNet(BasicBlock, [2, 2, 2, 2], color_channel=color_channel, num_classes=num_classes)

def nonetfResNet18(color_channel=3, num_classes=10):
    return nonetfResNet(BasicBlock, [2, 2, 2, 2], color_channel=color_channel, num_classes=num_classes)

#def ResNet34(color_channel=3, num_classes=10):
#    return ResNet(BasicBlock, [3, 4, 6, 3], color_channel=color_channel, num_classes=num_classes)
#
#
#def ResNet50(color_channel=3, num_classes=10):
#    return ResNet(Bottleneck, [3, 4, 6, 3], color_channel=color_channel, num_classes=num_classes)
#
#
#def ResNet101(color_channel=3, num_classes=10):
#    return ResNet(Bottleneck, [3, 4, 23, 3], color_channel=color_channel, num_classes=num_classes)
#
#
#def ResNet152(color_channel=3, num_classes=10):
#    return ResNet(Bottleneck, [3, 8, 36, 3], color_channel=color_channel, num_classes=num_classes)
#
#
#cfg = {
#    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
#    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
#    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
#    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
#}
#
#
#class VGG(nn.Module):
#    def __init__(self, vgg_name, color_channel=3, num_classes=10):
#        super(VGG, self).__init__()
#        self.color_channel = color_channel
#        self.features = self._make_layers(cfg[vgg_name])
#        self.fc1 = nn.Linear(512, 4096)
#        self.bn1 = nn.BatchNorm1d(4096)
#        self.fc2 = nn.Linear(4096, 4096)
#        self.bn2 = nn.BatchNorm1d(4096)
#        self.classifier = nn.Linear(4096, num_classes)
#
#    def _make_layers(self, cfg):
#        layers = []
#        in_channels = self.color_channel
#        for x in cfg:
#            if x == 'M':
#                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
#            else:
#                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
#                           nn.BatchNorm2d(x),
#                           nn.ReLU(inplace=True)]
#                in_channels = x
#        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
#        return nn.Sequential(*layers)
#
#    def forward(self, x):
#        out = self.features(x)
#        out = out.view(out.size(0), -1)
#        out = F.relu(self.bn1(self.fc1(out)))
#        out = F.relu(self.bn2(self.fc2(out)))
#        out = self.classifier(out)
#        return out
#
#    def get_features(self, x):
#        out = self.features(x)
#        out = out.view(out.size(0), -1)
#        out = F.relu(self.bn1(self.fc1(out)))
#        out = F.relu(self.bn2(self.fc2(out)))
#        return out
