import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt  # 用于绘制图像import pandas as pd  # 引入pandas库
import pandas as pd  # 引入pandas库

#nonetf

from utils import set_random_seed, get_minibatches_idx
#from models import ResNet18, VGG
from models import ResNet18, nonetfResNet18
from data import save_train_data, save_test_data, load_data_from_pickle

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.set_default_tensor_type('torch.cuda.FloatTensor')

from os import path, pardir
current_dir = path.abspath(path.dirname(__file__))
parent_dir = path.abspath(path.join(current_dir, pardir))

def build_model(config):
    if config['model'] == 'ResNet18':
        model = ResNet18(color_channel=config['color_channel'])
    elif config['model'] == 'nonetfResNet18':
        model = nonetfResNet18(color_channel=config['color_channel'])
    elif config['model'] == 'VGG11':
        model = VGG('VGG11', color_channel=config['color_channel'])
    elif config['model'] == 'VGG13':
        model = VGG('VGG13', color_channel=config['color_channel'])
    else:
        print('wrong model option')
        model = None
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config['lr'],  momentum=config['momentum'],
                          weight_decay=config['weight_decay'])

    return model, loss_function, optimizer


def simple_train_batch(trainloader, model, loss_function, optimizer, config):
    model.train()
    losses = []  # 初始化一个空列表，用于存储每一轮训练的平均损失
    accuracies = []  # 初始化一个空列表，用于存储每一轮训练的准确率
    for epoch in range(config['epoch_num']):
        if epoch == int(config['epoch_num'] / 3):
            for g in optimizer.param_groups:
                g['lr'] = config['lr'] / 10
            print('divide current learning rate by 10')
        elif epoch == int(config['epoch_num'] * 2 / 3):
            for g in optimizer.param_groups:
                g['lr'] = config['lr'] / 100
            print('divide current learning rate by 10')
        total_loss = 0
        total = 0.0
        correct = 0.0
        minibatches_idx = get_minibatches_idx(len(trainloader), minibatch_size=config['simple_train_batch_size'],
                                              shuffle=True)
        for minibatch in minibatches_idx:
            inputs = torch.Tensor(np.array([list(trainloader[x][0].cpu().numpy()) for x in minibatch]))
            targets = torch.Tensor(np.array([list(trainloader[x][1].cpu().numpy()) for x in minibatch]))
            inputs, targets = Variable(inputs.cuda()).squeeze(1), Variable(targets.long().cuda()).squeeze()
            optimizer.zero_grad()
            outputs = model(inputs).squeeze()
            loss = loss_function(outputs, targets)
            total_loss += loss
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.long()).sum().item()
            loss.backward()
            optimizer.step()
        avg_loss = total_loss / len(minibatches_idx)
        avg_acc = correct / total  # 计算平均准确率
        losses.append(avg_loss.item())
        accuracies.append(avg_acc)  # 将每一轮的平均准确率添加到accuracies列表中
        print('epoch:', epoch, 'loss:', total_loss.item(), 'accuracy:', avg_acc)
    return losses, accuracies


def simple_test_batch(testloader, model, config):
    model.eval()
    total = 0.0
    correct = 0.0
    minibatches_idx = get_minibatches_idx(len(testloader), minibatch_size=config['simple_test_batch_size'],
                                          shuffle=False)
    y_true = []
    y_pred = []
    for minibatch in minibatches_idx:
        inputs = torch.Tensor(np.array([list(testloader[x][0].cpu().numpy()) for x in minibatch]))
        targets = torch.Tensor(np.array([list(testloader[x][1].cpu().numpy()) for x in minibatch]))
        inputs, targets = Variable(inputs.cuda()).squeeze(1), Variable(targets.cuda()).squeeze()
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.long()).sum().item()
        y_true.extend(targets.cpu().data.numpy().tolist())
        y_pred.extend(predicted.cpu().data.numpy().tolist())
    test_accuracy = correct / total
    test_confusion_matrix = confusion_matrix(y_true, y_pred)
    t1 = config['t1']
    big_class_acc = np.sum([test_confusion_matrix[i, i] for i in range(t1)]) / np.sum(test_confusion_matrix[:t1])
    if t1 == 10:
        small_class_acc = None
    else:
        small_class_acc = \
            np.sum([test_confusion_matrix[i, i] for i in range(10)[t1:]]) / np.sum(test_confusion_matrix[t1:])
    return test_accuracy, big_class_acc, small_class_acc, test_confusion_matrix


def run_train_models():
    data_option = sys.argv[1].split('=')[1]
    model_option = sys.argv[2].split('=')[1]
    t1 = int(sys.argv[3].split('=')[1])
    R = sys.argv[4].split('=')[1]
    config = {'dir_path': current_dir,
              'data': data_option,
              'model': model_option,
              't1': t1,
              'R': R,
              'simple_train_batch_size': 128,
              'simple_test_batch_size': 100,
              'epoch_num': 40,
              'lr': 0.1,
              'momentum': 0.9,
              'weight_decay': 5e-4,
              'fixed': 'big'}

    # fixed: big/small
    if data_option == 'fashion_mnist':
        config['color_channel'] = 1
    else:
        config['color_channel'] = 3
    if R == 'inf':
        config['big_class_sample_size'] = 5000
        config['small_class_sample_size'] = 0
    else:
        R = int(R)
        if data_option == 'cifar10':
            config['big_class_sample_size'] = 5000
            config['small_class_sample_size'] = 5000 // R
        elif data_option == 'fashion_mnist':
            config['big_class_sample_size'] = 6000
            config['small_class_sample_size'] = 6000 // R
        else:
            print('wrong data option')
    model_path = config['dir_path'] + '/models/' + config['data'] + '_' + config['model'] + '_t1=' + \
                 str(config['t1']) + '_R=' + config['R'] + "_" + config['fixed'] + '.pt'

    print('save test data')
    set_random_seed(666)
    save_test_data(config)

    print('save train data')
    set_random_seed(666)
    save_train_data(config)

    set_random_seed(666)
    print('load data from pickle')
    train_data, test_data = load_data_from_pickle(config)

    print('build model')
    model, loss_function, optimizer = build_model(config)
    print('train model')
    losses, accuracies = simple_train_batch(train_data, model, loss_function, optimizer, config)
    print('save model')
    torch.save(model.state_dict(), model_path)
    print('load model')
    model.load_state_dict(torch.load(model_path))
    train_res, train_big, train_small, train_confusion_matrix = simple_test_batch(train_data, model, config)
    test_res, test_big, test_small, test_confusion_matrix = simple_test_batch(test_data, model, config)
    print('train accuracy', train_res, train_big, train_small)
    print('test accuracy', test_res, test_big, test_small)
    print('train confusion matrix\n', train_confusion_matrix)
    print('test confusion matrix\n', test_confusion_matrix)

    # 将损失和准确率保存为CSV文件
    df = pd.DataFrame({
        'epoch': range(1, config['epoch_num'] + 1),
        'loss': losses,
        'accuracy': accuracies
    })
    df.to_csv(config['dir_path'] + '/training_metrics.csv', index=False)
    
    # 绘制损失和准确率曲线并保存为图片
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(losses, label='Train Loss')
    plt.title('Loss over epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    plt.subplot(1, 2, 2)
    plt.plot(accuracies, label='Train Accuracy')
    plt.title('Accuracy over epochs')
    plt.xlabel('Epochs')


if __name__ == '__main__':
    run_train_models()
