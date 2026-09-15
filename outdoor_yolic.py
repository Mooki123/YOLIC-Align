#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import random

from PIL import Image
from torch.optim.lr_scheduler import StepLR, MultiStepLR
import argparse
import numpy as np
import torch
import torchvision
from torchvision import models
from torchvision import transforms, datasets
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR
import torch.nn as nn
import time
import copy
import cv2
import os.path
import matplotlib.pyplot as plt
import pandas as pd
import os
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler as GradScaler
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights, ShuffleNet_V2_X1_0_Weights

from shufflenet import shufflenet_v2_x1_0

parser = argparse.ArgumentParser(description='PyTorch Training Script')
parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test_batch', type=int, default=32, metavar='N',
                    help='input batch size for testing (default: 64)')
parser.add_argument('--epochs', type=int, default=150, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=25, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--resume', type=bool, default=True, metavar='N',
                    help='resume from the last weights')

NumCell = 104  # number of cells
NumClass = 11  # number of classes
save_name = 'mobilenet_outdoor'  # name of the model

points_list = [(288, 166), (322, 166), (356, 166), (390, 166), (424, 166), (458, 166), (492, 166), (526, 166),
               (220, 200), (254, 200), (288, 200), (322, 200), (356, 200), (390, 200), (424, 200), (458, 200),
               (492, 200), (526, 200), (560, 200), (594, 200),
               (220, 234), (254, 234), (288, 234), (322, 234), (356, 234), (390, 234), (424, 234), (458, 234),
               (492, 234), (526, 234), (560, 234), (594, 234), (628, 234),
               (254, 268), (288, 268), (322, 268), (356, 268), (390, 268), (424, 268), (458, 268), (492, 268),
               (526, 268), (560, 268), (594, 268), (628, 268),
               (0, 268), (53, 268), (106, 268), (159, 268), (212, 268), (265, 268), (318, 268), (371, 268), (424, 268),
               (477, 268), (530, 268), (583, 268), (636, 268), (689, 268), (742, 268), (795, 268),
               (0, 321), (53, 321), (106, 321), (159, 321), (212, 321), (265, 321), (318, 321), (371, 321), (424, 321),
               (477, 321), (530, 321), (583, 321), (636, 321), (689, 321), (742, 321), (795, 321), (848, 321),
               (0, 374), (53, 374), (106, 374), (159, 374), (212, 374), (265, 374), (318, 374), (371, 374), (424, 374),
               (477, 374), (530, 374), (583, 374), (636, 374), (689, 374), (742, 374), (795, 374), (848, 374),
               (0, 427), (53, 427), (106, 427), (159, 427), (212, 427), (265, 427), (318, 427), (371, 427), (424, 427),
               (477, 427), (530, 427), (583, 427), (636, 427), (689, 427), (742, 427), (795, 427), (848, 427),
               (0, 480), (53, 480), (106, 480), (159, 480), (212, 480), (265, 480), (318, 480), (371, 480), (424, 480),
               (477, 480), (530, 480), (583, 480), (636, 480), (689, 480), (742, 480), (795, 480), (848, 480),
               (184, 0), (244, 0), (304, 0), (364, 0), (424, 0), (484, 0), (544, 0), (604, 0), (244, 60), (304, 60),
               (364, 60), (424, 60), (484, 60), (544, 60), (604, 60), (664, 60)]
cell_list = [[points_list[0], points_list[11]], [points_list[1], points_list[12]], [points_list[2], points_list[13]],
            [points_list[3], points_list[14]], [points_list[4], points_list[15]],
            [points_list[5], points_list[16]], [points_list[6], points_list[17]], [points_list[7], points_list[18]],
            [points_list[8], points_list[21]], [points_list[9], points_list[22]],
            [points_list[10], points_list[23]], [points_list[11], points_list[24]], [points_list[12], points_list[25]],
            [points_list[13], points_list[26]], [points_list[14], points_list[27]],
            [points_list[15], points_list[28]], [points_list[16], points_list[29]], [points_list[17], points_list[30]],
            [points_list[18], points_list[31]], [points_list[19], points_list[32]],
            [points_list[20], points_list[33]], [points_list[21], points_list[34]], [points_list[22], points_list[35]],
            [points_list[23], points_list[36]], [points_list[24], points_list[37]],
            [points_list[25], points_list[38]], [points_list[26], points_list[39]], [points_list[27], points_list[40]],
            [points_list[28], points_list[41]], [points_list[29], points_list[42]],
            [points_list[30], points_list[43]], [points_list[31], points_list[44]], [points_list[45], points_list[62]],
            [points_list[46], points_list[63]], [points_list[47], points_list[64]],
            [points_list[48], points_list[65]], [points_list[49], points_list[66]], [points_list[50], points_list[67]],
            [points_list[51], points_list[68]], [points_list[52], points_list[69]],
            [points_list[53], points_list[70]], [points_list[54], points_list[71]], [points_list[55], points_list[72]],
            [points_list[56], points_list[73]], [points_list[57], points_list[74]],
            [points_list[58], points_list[75]], [points_list[59], points_list[76]], [points_list[60], points_list[77]],
            [points_list[61], points_list[79]], [points_list[62], points_list[80]],
            [points_list[63], points_list[81]], [points_list[64], points_list[82]], [points_list[65], points_list[83]],
            [points_list[66], points_list[84]], [points_list[67], points_list[85]],
            [points_list[68], points_list[86]], [points_list[69], points_list[87]], [points_list[70], points_list[88]],
            [points_list[71], points_list[89]], [points_list[72], points_list[90]],
            [points_list[73], points_list[91]], [points_list[74], points_list[92]], [points_list[75], points_list[93]],
            [points_list[76], points_list[94]], [points_list[78], points_list[96]],
            [points_list[79], points_list[97]], [points_list[80], points_list[98]], [points_list[81], points_list[99]],
            [points_list[82], points_list[100]], [points_list[83], points_list[101]],
            [points_list[84], points_list[102]], [points_list[85], points_list[103]],
            [points_list[86], points_list[104]], [points_list[87], points_list[105]],
            [points_list[88], points_list[106]],
            [points_list[89], points_list[107]], [points_list[90], points_list[108]],
            [points_list[91], points_list[109]], [points_list[92], points_list[110]],
            [points_list[93], points_list[111]],
            [points_list[95], points_list[113]], [points_list[96], points_list[114]],
            [points_list[97], points_list[115]], [points_list[98], points_list[116]],
            [points_list[99], points_list[117]],
            [points_list[100], points_list[118]], [points_list[101], points_list[119]],
            [points_list[102], points_list[120]], [points_list[103], points_list[121]],
            [points_list[104], points_list[122]],
            [points_list[105], points_list[123]], [points_list[106], points_list[124]],
            [points_list[107], points_list[125]], [points_list[108], points_list[126]],
            [points_list[109], points_list[127]],
            [points_list[110], points_list[128]], [points_list[129], points_list[137]],
            [points_list[130], points_list[138]], [points_list[131], points_list[139]],
            [points_list[132], points_list[140]],
            [points_list[133], points_list[141]], [points_list[134], points_list[142]],
            [points_list[135], points_list[143]], [points_list[136], points_list[144]]]

def generate_fractional_masks(cell_list, original_size=(848, 480), input_size=(224, 224), feature_size=(7, 7)):
    import cv2
    import numpy as np
    
    masks = []
    scale_x = input_size[0] / original_size[0]
    scale_y = input_size[1] / original_size[1]
    
    for cell in cell_list:
        x1 = int(cell[0][0] * scale_x)
        y1 = int(cell[0][1] * scale_y)
        x2 = int(cell[1][0] * scale_x)
        y2 = int(cell[1][1] * scale_y)
        
        mask = np.zeros((input_size[1], input_size[0]), dtype=np.float32)
        cv2.rectangle(mask, (x1, y1), (x2, y2), 1.0, -1)
        masks.append(mask)
        
    mask_tensor = torch.tensor(np.stack(masks))
    mask_tensor = mask_tensor.unsqueeze(0)
    mask_features = F.adaptive_avg_pool2d(mask_tensor, feature_size)
    mask_features = mask_features.squeeze(0)
    return mask_features

class YolicAlignModel(nn.Module):
    def __init__(self, pretrained=True):
        super(YolicAlignModel, self).__init__()
        weights = MobileNet_V2_Weights.DEFAULT if pretrained else None
        base_model = mobilenet_v2(weights=weights)
        self.features = base_model.features
        
        self.classifier = nn.Linear(1280, NumClass + 1)
        self.register_buffer('fractional_masks', generate_fractional_masks(cell_list, original_size=(848, 480)))
        
    def forward(self, x):
        feat = self.features(x)
        B = feat.shape[0]
        
        feat_expanded = feat.unsqueeze(2) 
        masks_expanded = self.fractional_masks.unsqueeze(0).unsqueeze(0)
        
        masked_feat = feat_expanded * masks_expanded
        pooled = masked_feat.sum(dim=(3, 4))
        
        mask_area = masks_expanded.sum(dim=(3, 4))
        pooled = pooled / (mask_area + 1e-6)
        
        pooled = pooled.transpose(1, 2)
        logits = self.classifier(pooled)
        logits = logits.reshape(B, -1)
        return logits

model = YolicAlignModel(pretrained=True)
optimizer = optim.Adam(model.parameters(), lr=0.001)  # optimizer and learning rate
torch.cuda.empty_cache()
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


def random_augmentation(image, label_list, seq_list):
    # flip image horizontally
    image = image.flip(1)
    n_groups = len(seq_list)
    n_labels = len(label_list)
    assert n_labels % n_groups == 0  # make sure it's evenly divisible

    group_size = n_labels // n_groups

    # divide the label_list into groups based on seq_list
    label_groups = []
    start_idx = 0
    for group_idx in seq_list:
        end_idx = start_idx + group_size
        label_groups.append(label_list[start_idx:end_idx])
        start_idx = end_idx

    # create a new label_list based on seq_list
    new_label_list = []
    for group_idx in seq_list:
        group = label_groups[group_idx]
        new_label_list.extend(group)

    return image, new_label_list


class MultiLabelRGBataSet(torch.utils.data.Dataset):
    def __init__(self, imgspath, imgslist, annotationpath, transforms=None):
        self.imgslist = imgslist
        self.imgspath = imgspath
        self.transform = transforms
        self.annotationpath = annotationpath
        # print(annotationpath)

    def __len__(self):
        return len(self.imgslist)

    def __getitem__(self, index):
        ipath = os.path.join(self.imgspath, self.imgslist[index])
        img = Image.open(ipath)
        if self.transform is not None:
            img = self.transform(img)
        (filename, extension) = os.path.splitext(ipath)
        filename = os.path.basename(filename)
        annotation = os.path.join(self.annotationpath, filename + ".txt")
        label = np.loadtxt(annotation, dtype=np.int64)
        if random.random() > 0.5:
            img, label = random_augmentation(img, label, [7, 6, 5, 4, 3, 2, 1, 0, 19, 18, 17, 16, 15, 14, 13, 12, 11,
                                                          10, 9, 8, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 47,
                                                          46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32,
                                                          63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49,
                                                          48, 79, 78, 77, 76, 75, 74, 73, 72, 71, 70, 69, 68, 67, 66,
                                                          65, 64, 95, 94, 93, 92, 91, 90, 89, 88, 87, 86, 85, 84, 83,
                                                          82, 81, 80, 103, 102, 101, 100, 99, 98, 97, 96])
        label = torch.tensor(label, dtype=torch.float32)
        return img, label, filename


train_trans = transforms.Compose(([

    transforms.Resize((224, 224)),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
    transforms.ToTensor()  # divides by 255
]))
val_test_trans = transforms.Compose(([
    transforms.Resize((224, 224)),
    transforms.ToTensor()  # divides by 255
]))

img_dir = 'images'
label_dir = 'yoliclabel'
img_list = os.listdir(img_dir)
train_img, Val_Test = train_test_split(img_list, test_size=0.3, random_state=2)
val_img, test_img = train_test_split(Val_Test, test_size=0.6666, random_state=2)

train = MultiLabelRGBataSet(img_dir, train_img, label_dir, train_trans)
valid = MultiLabelRGBataSet(img_dir, val_img, label_dir, val_test_trans)
test = MultiLabelRGBataSet(img_dir, test_img, label_dir, val_test_trans)

train_loader = torch.utils.data.DataLoader(train,
                                           batch_size=args.batch_size,
                                           shuffle=True, num_workers=8)
valid_loader = torch.utils.data.DataLoader(valid,
                                           batch_size=args.batch_size,
                                           shuffle=False, num_workers=8)

test_loader = torch.utils.data.DataLoader(test,
                                          batch_size=args.batch_size,
                                          shuffle=False, num_workers=8)

if args.cuda:
    model.cuda()

criterion = nn.BCEWithLogitsLoss()
scheduler = MultiStepLR(optimizer, milestones=[100, 125], gamma=0.1)


def pred_acc(original, predicted):
    pred = torch.round(predicted).detach().numpy().astype(np.int64)
    orig = original.detach().numpy()
    pred = np.reshape(pred, (NumCell * (NumClass + 1), 1)).flatten()
    orig = np.reshape(orig, (NumCell * (NumClass + 1), 1)).flatten()
    num = 0
    enum = 0
    normal = np.asarray([0] * NumClass + [1])
    for cell in range(0, (NumCell * (NumClass + 1)), NumClass + 1):
        if (orig[cell:cell + NumClass + 1] == pred[cell:cell + NumClass + 1]).all():
            num = num + 1
        else:
            if not (orig[cell:cell + NumClass + 1] == normal).all() and not (
                    pred[cell:cell + NumClass + 1] == normal).all():
                enum = enum + 1
    return num / NumCell, (num + enum) / NumCell


def train(epoch, model):
    model.train()
    for batch_idx, (data, target, filenames) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        target = target.type_as(output)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.5f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


best_correct = -999


def evaluate(model, data_loader, save_mode=False):
    model.eval()
    running_loss = []
    running_acc = []
    running_binary = []
    global best_correct
    with torch.no_grad():
        for batch_idx, (data, target, filenames) in enumerate(data_loader):
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            output = model(data)
            target = target.type_as(output)
            loss = criterion(output, target)
            output = torch.sigmoid(output)
            acc_all = []
            acc_binary = []
            for each_image, d in enumerate(output):
                all_acc, b_acc = pred_acc(torch.Tensor.cpu(target[each_image]), torch.Tensor.cpu(d))
                acc_all.append(all_acc)
                acc_binary.append(b_acc)
            running_loss.append(loss.item())
            running_acc.append(np.asarray(acc_all).mean())
            running_binary.append(np.asarray(acc_binary).mean())
    total_batch_loss = np.asarray(running_loss).mean()
    total_batch_acc = np.asarray(running_acc).mean()
    total_batch_binary = np.asarray(running_binary).mean()
    print('\n loader set: total_batch_loss: {:.4f}, total imgs: {} , Acc: ({:.4f}%), Binary ACC: ({:.4f}%)\n'.format(
        total_batch_loss, len(data_loader.dataset), total_batch_acc, total_batch_binary))
    if save_mode:
        now_correct = total_batch_acc
        if best_correct < now_correct:
            best_correct = now_correct
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts,
                       os.path.join(os.getcwd(), save_name + ".pth.tar"))
            print("New weight!")
    return total_batch_loss, total_batch_acc


if __name__ == '__main__':
    # test_loss, test_acc = test(model)
    import datetime

    start_time = datetime.datetime.now()
    print(save_name)
    all_train_loss = []
    all_train_acc = []
    all_val_loss = []
    all_val_acc = []
    all_test_loss = []
    all_test_acc = []
    for epoch in range(1, args.epochs + 1):
        train(epoch, model)
        train_loss, train_acc = evaluate(model, train_loader)
        val_loss, val_acc = evaluate(model, valid_loader, save_mode=True)
        test_loss, test_acc = evaluate(model, test_loader)
        all_train_acc.append(train_acc)
        all_train_loss.append(train_loss)
        all_val_acc.append(val_acc)
        all_val_loss.append(val_loss)
        all_test_loss.append(test_loss)
        all_test_acc.append(test_acc)
        scheduler.step()
    list_res = []
    for i in range(len(all_train_loss)):
        list_res.append([all_train_loss[i], all_train_acc[i], all_val_loss[i], all_val_acc[i],
                         all_test_loss[i], all_test_acc[i]])

    column_name = ['train_loss', 'train_acc', 'val_loss', 'val_acc', 'test_loss', 'test_acc']
    csv_name = save_name + '.csv'
    xml_df = pd.DataFrame(list_res, columns=column_name)
    xml_df.to_csv(csv_name, index=None)
    end_time = datetime.datetime.now()
    print('\nTime taken: {}\n'.format(end_time - start_time))
